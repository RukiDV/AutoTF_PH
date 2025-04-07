#include "gpu_renderer.hpp"

#include "event_handler.hpp"
#include "work_context.hpp"
#include "util/timer.hpp"
#include "SDL3/SDL_mouse.h"

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>

struct GPUContext 
{
    GPUContext(AppState &app_state, const Volume &volume)
        : vcc(vmc), wc(vmc, vcc)
    {
        vmc.construct(app_state.get_window_extent().width, app_state.get_window_extent().height);
        vcc.construct();
        wc.construct(app_state, volume);
    }

    ~GPUContext() 
    {
        wc.destruct();
        vcc.destruct();
        vmc.destruct();
    }

    ve::VulkanMainContext vmc;
    ve::VulkanCommandContext vcc;
    ve::WorkContext wc;
};

void dispatch_pressed_keys(GPUContext &gpu_context, EventHandler &eh, AppState &app_state) 
{
    float move_amount = app_state.time_diff * app_state.move_speed;
    if (eh.is_key_pressed(Key::W)) app_state.cam.move_front(move_amount);
    if (eh.is_key_pressed(Key::A)) app_state.cam.move_right(-move_amount);
    if (eh.is_key_pressed(Key::S)) app_state.cam.move_front(-move_amount);
    if (eh.is_key_pressed(Key::D)) app_state.cam.move_right(move_amount);
    if (eh.is_key_pressed(Key::Q)) app_state.cam.move_up(-move_amount);
    if (eh.is_key_pressed(Key::E)) app_state.cam.move_up(move_amount);
    float panning_speed = eh.is_key_pressed(Key::Shift) ? 50.0f : 200.0f;
    if (eh.is_key_pressed(Key::Left)) app_state.cam.on_mouse_move(glm::vec2(-panning_speed * app_state.time_diff, 0.0f));
    if (eh.is_key_pressed(Key::Right)) app_state.cam.on_mouse_move(glm::vec2(panning_speed * app_state.time_diff, 0.0f));
    if (eh.is_key_pressed(Key::Up)) app_state.cam.on_mouse_move(glm::vec2(0.0f, -panning_speed * app_state.time_diff));
    if (eh.is_key_pressed(Key::Down)) app_state.cam.on_mouse_move(glm::vec2(0.0f, panning_speed * app_state.time_diff));

    if (eh.is_key_released(Key::Plus)) 
    {
        app_state.move_speed *= 2.0f;
        eh.set_released_key(Key::Plus, false);
    }
    if (eh.is_key_released(Key::Minus)) 
    {
        app_state.move_speed /= 2.0f;
        eh.set_released_key(Key::Minus, false);
    }
    if (eh.is_key_released(Key::G)) 
    {
        app_state.show_ui = !app_state.show_ui;
        eh.set_released_key(Key::G, false);
    }
    if (eh.is_key_released(Key::F1)) 
    {
        app_state.save_screenshot = true;
        eh.set_released_key(Key::F1, false);
    }
    if (eh.is_key_pressed(Key::MouseLeft)) 
    {
        if (!SDL_GetWindowRelativeMouseMode(gpu_context.vmc.window->get())) 
        {
            SDL_SetWindowRelativeMouseMode(gpu_context.vmc.window->get(), true);
        }
        app_state.cam.on_mouse_move(glm::vec2(eh.mouse_motion.x * 1.5f, eh.mouse_motion.y * 1.5f));
        eh.mouse_motion = glm::vec2(0.0f);
    }
    if (eh.is_key_released(Key::MouseLeft)) 
    {
        SDL_SetWindowRelativeMouseMode(gpu_context.vmc.window->get(), false);
        SDL_WarpMouseInWindow(gpu_context.vmc.window->get(), app_state.get_window_extent().width / 2.0f, app_state.get_window_extent().height / 2.0f);
        eh.set_released_key(Key::MouseLeft, false);
    }
}

std::vector<PersistencePair> calculate_persistence_pairs(const Volume &volume, std::vector<int>& filtration_values, FiltrationMode mode)
{
    auto [boundary_matrix, filt_vals] = create_boundary_matrix(volume, mode);
    filtration_values = filt_vals;
    std::vector<PersistencePair> raw_pairs = boundary_matrix.reduce();
    return raw_pairs;
}

void print_merge_tree(MergeTreeNode *node, int level = 0)
{
    if (!node)
        return;
    std::string indent(level * 2, ' ');
    std::cout << indent << "Node ID=" << node->id << ", Birth=" << node->birth << ", Death=" << node->death << ", Depth=" << node->depth;
    if (node->parent)
        std::cout << ", Parent=" << node->parent->id;
    else
        std::cout << " (root)";
    std::cout << std::endl;
    for (auto child : node->children)
    {
        print_merge_tree(child, level + 1);
    }
}

// export merge tree edges to a file (each line: parent child)
void exportMergeTreeEdges(const MergeTree &merge_tree, const std::string &filename)
{
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << " for writing." << std::endl;
        return;
    }
    
    const auto &nodes = merge_tree.get_all_nodes();
    for (const auto &pair : nodes)
    {
        MergeTreeNode* node = pair.second;
        for (auto child : node->children)
        {
            ofs << node->id << " " << child->id << "\n";
        }
    }
    ofs.close();
    std::cout << "Merge tree edges exported to " << filename << std::endl;
}

// export merge tree edges to a file, filtering out nodes deeper than maxDepth and only exporting edges where the child's persistence meets the minimum threshold
void exportFilteredMergeTreeEdges(const MergeTree &merge_tree, const std::string &filename, int maxDepth, int minPersistence)
{
    std::ofstream ofs(filename);
    if (!ofs.is_open())
    {
        std::cerr << "Error: Cannot open file " << filename << " for writing." << std::endl;
        return;
    }
    
    // determine which nodes are within the allowed depth
    const auto &nodes = merge_tree.get_all_nodes();
    std::unordered_set<uint32_t> filteredNodeIds;
    for (const auto &pair : nodes) {
        MergeTreeNode* node = pair.second;
        if (node->depth <= maxDepth)
        {
            filteredNodeIds.insert(node->id);
        }
    }
    
    // export only edges where both the parent and the child are in the filtered set, and the child's persistence meets the minPersistence threshold
    for (const auto &pair : nodes) 
    {
        MergeTreeNode* parent = pair.second;
        if (filteredNodeIds.find(parent->id) == filteredNodeIds.end())
            continue;
        for (auto child : parent->children)
        {
            if (filteredNodeIds.find(child->id) != filteredNodeIds.end())
            {
                int persistence = child->death - child->birth;
                if (persistence >= minPersistence)
                {
                    ofs << parent->id << " " << child->id << "\n";
                }
            }
        }
    }
    
    ofs.close();
    std::cout << "Filtered merge tree edges exported to " << filename << " (max depth = " << maxDepth << ", min persistence = " << minPersistence << ")" << std::endl;
}

// collect nodes at a given target level recursively
void get_nodes_at_level(MergeTreeNode *node, int currentLevel, int targetLevel, std::vector<MergeTreeNode *> &result) 
{
    if (!node)
        return;
    if (currentLevel == targetLevel) 
    {
        result.push_back(node);
        return;
    }
    for (auto child : node->children) 
    {
        get_nodes_at_level(child, currentLevel + 1, targetLevel, result);
    }
}

// return persistence pairs corresponding to nodes at a given target level
std::vector<PersistencePair> get_persistence_pairs_for_level(MergeTree& merge_tree, int targetLevel) 
{
    std::vector<MergeTreeNode*> levelNodes;
    auto all_nodes = merge_tree.get_all_nodes();
    for (const auto& [id, node] : all_nodes) 
    {
        if (node->parent == nullptr) 
        {
            get_nodes_at_level(node, 0, targetLevel, levelNodes);
        }
    }
    std::cout << "Total nodes found at target level " << targetLevel << ": " << levelNodes.size() << std::endl;
    
    std::vector<PersistencePair> selectedPairs;
    for (auto n : levelNodes) 
    {
        std::cout << "Selected PersistencePair: Birth = " << n->birth << ", Death = " << n->death << ", Depth = " << n->depth << std::endl;
        selectedPairs.push_back(PersistencePair(n->birth, n->death));
    }
    return selectedPairs;
}

void debug_print_merge_tree(const MergeTree &merge_tree) 
{
    std::cout << "=== Merge Tree Debug Info ===" << std::endl;
    int max_depth = 0;
    const auto &nodes = merge_tree.get_all_nodes();
    for (const auto &pair : nodes) {
        const auto *node = pair.second;
        std::cout << "Node " << pair.first << ": birth=" << node->birth << ", death=" << node->death << ", depth=" << node->depth;
        if (node->parent)
            std::cout << ", parent=" << node->parent->id;
        else
            std::cout << " (root)";
        std::cout << std::endl;
        if (node->depth > max_depth)
            max_depth = node->depth;
    }
    std::cout << "Maximum merge tree depth: " << max_depth << std::endl;
}

void debug_print_nodes_at_level(const MergeTree &merge_tree, int targetLevel) 
{
    std::vector<MergeTreeNode *> levelNodes;
    const auto &nodes = merge_tree.get_all_nodes();
    for (const auto &pair : nodes) {
        MergeTreeNode *node = pair.second;
        if (node->parent == nullptr) {
            get_nodes_at_level(node, 0, targetLevel, levelNodes);
        }
    }
    std::cout << "Nodes at target level " << targetLevel << ":" << std::endl;
    for (const auto *node : levelNodes) {
        std::cout << "  Node " << node->id << " (birth=" << node->birth << ", death=" << node->death << ", depth=" << node->depth << ")" << std::endl;
    }
}

int gpu_render(const Volume &volume) 
{
    AppState app_state;
    EventHandler eh;
    GPUContext gpu_context(app_state, volume);

    std::vector<int> filtration_values;
    std::vector<PersistencePair> raw_pairs = calculate_persistence_pairs(volume, filtration_values, app_state.filtration_mode);
    std::cout << "Raw persistence pairs: " << raw_pairs.size() << std::endl;
    
    gpu_context.wc.getMergeTree() = build_merge_tree_with_tolerance(raw_pairs, 5);
    exportFilteredMergeTreeEdges(gpu_context.wc.getMergeTree(), "merge_tree_edges_filtered.txt", 3, 10);

    std::ofstream outfile("persistence_pairs.txt");
    if (outfile.is_open()) {
        for (const auto &pair : raw_pairs) {
            outfile << filtration_values[pair.birth] << " " << filtration_values[pair.death] << "\n";
        }
        outfile.close();
        std::cout << "Persistence pairs saved to persistence_pairs.txt" << std::endl;
    } else {
        std::cerr << "Failed to open persistence_pairs.txt for writing!" << std::endl;
    }

    std::string outputFile = "output_plots/persistence_diagram.png";
    std::string pythonCommand = "python scripts/persistence_diagram.py persistence_pairs.txt " + outputFile;
    int ret = system(pythonCommand.c_str());
    if (ret != 0) 
    {
         std::cerr << "Python script for persistence diagram failed with error code " << ret << std::endl;
    } else {
         std::cout << "Persistence diagram generated successfully." << std::endl;
    }

    ImTextureID persistenceTex = gpu_context.wc.load_persistence_diagram_texture(outputFile);
    gpu_context.wc.set_ui_persistence_texture(persistenceTex);

    bool quit = false;
    Timer rendering_timer;
    SDL_Event e;
    while (!quit)
    {
        dispatch_pressed_keys(gpu_context, eh, app_state);
        app_state.cam.update();

        if (app_state.apply_filtration_mode)
        {
            raw_pairs = calculate_persistence_pairs(volume, filtration_values, app_state.filtration_mode);
            std::cout << "Filtration mode updated. New raw persistence pairs: " << raw_pairs.size() << std::endl;
            gpu_context.wc.getMergeTree() = build_merge_tree_with_tolerance(raw_pairs, 5);
            app_state.apply_filtration_mode = false;
        }

        if (app_state.apply_persistence_threshold)
        {
            std::vector<PersistencePair> currentFilteredPairs = threshold_cut(raw_pairs, app_state.persistence_threshold);
            std::cout << "Persistence threshold updated to " << app_state.persistence_threshold << ", filtered pairs: " << currentFilteredPairs.size() << std::endl;
            gpu_context.wc.getMergeTree() = build_merge_tree_with_tolerance(currentFilteredPairs, 2);
            std::vector<PersistencePair> selectedPairs = get_persistence_pairs_for_level(gpu_context.wc.getMergeTree(), app_state.target_level);
            gpu_context.wc.set_persistence_pairs(selectedPairs, volume);
            app_state.apply_persistence_threshold = false;
        }

        if (app_state.apply_target_level)
        {
            std::vector<PersistencePair> selectedPairs = get_persistence_pairs_for_level(gpu_context.wc.getMergeTree(), app_state.target_level);
            gpu_context.wc.set_persistence_pairs(selectedPairs, volume);
            app_state.apply_target_level = false;
        }
        try
        {
            gpu_context.wc.draw_frame(app_state);
        } catch (const vk::OutOfDateKHRError &ex)
        {
            app_state.set_window_extent(gpu_context.wc.recreate_swapchain(app_state.vsync));
        }

        while (SDL_PollEvent(&e))
        {
            if (e.window.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED)
                quit = true;
            eh.dispatch_event(e);
        }
        app_state.time_diff = rendering_timer.restart();
    }
    return 0;
}

