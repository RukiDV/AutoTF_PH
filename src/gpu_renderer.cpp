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
#include <filesystem>

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
        //std::cout << "Selected PersistencePair: Birth = " << n->birth << ", Death = " << n->death << ", Depth = " << n->depth << std::endl;
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

    // decide on a volume‐specific cache path, hash the dimensions
    std::string cache_base = "cache/";
    std::string vol_id = std::to_string(volume.resolution.x) + "x" + std::to_string(volume.resolution.y) + "x" + std::to_string(volume.resolution.z);
    std::string pairs_cache = cache_base + vol_id + "_pairs.bin";
    std::string filt_cache  = cache_base + vol_id + "_filts.bin";
    std::vector<PersistencePair> raw_pairs;
    std::vector<int> filtration_values;

    // if the cache exists, load it
    if (std::filesystem::exists(pairs_cache) && std::filesystem::exists(filt_cache))
    {
        // load binary cache
        {
            std::ifstream in(pairs_cache, std::ios::binary);
            size_t N;
            in.read((char*)&N, sizeof(N));
            raw_pairs.resize(N);
            in.read((char*)raw_pairs.data(), sizeof(PersistencePair)*N);
        }
        {
            std::ifstream in(filt_cache, std::ios::binary);
            size_t M;
            in.read((char*)&M, sizeof(M));
            filtration_values.resize(M);
            in.read((char*)filtration_values.data(), sizeof(int)*M);
        }
        std::cout << "Loaded " << raw_pairs.size() << " persistence pairs from cache.\n";
    }
    else
    {
      // do the expensive compute, then write it out for next time
      raw_pairs = calculate_persistence_pairs(volume, filtration_values, app_state.filtration_mode);
      std::filesystem::create_directories(cache_base);
      {
        std::ofstream out(pairs_cache, std::ios::binary);
        size_t N = raw_pairs.size();
        out.write((char*)&N, sizeof(N));
        out.write((char*)raw_pairs.data(), sizeof(PersistencePair)*N);
      }
      {
        std::ofstream out(filt_cache, std::ios::binary);
        size_t M = filtration_values.size();
        out.write((char*)&M, sizeof(M));
        out.write((char*)filtration_values.data(), sizeof(int)*M);
      }
      std::cout << "Computed and cached " << raw_pairs.size() << " persistence pairs.\n";
    }

    // gradient
    std::string grad_pairs_cache = cache_base + vol_id + "_grad_pairs.bin";
    std::string grad_filt_cache  = cache_base + vol_id + "_grad_filts.bin";
    std::vector<PersistencePair> raw_grad_pairs;
    std::vector<int> grad_filtration_values;

    if (std::filesystem::exists(grad_pairs_cache) && std::filesystem::exists(grad_filt_cache))
    {
        std::ifstream inG(grad_pairs_cache, std::ios::binary);
        size_t G;
        inG.read((char*)&G, sizeof(G));
        raw_grad_pairs.resize(G);
        inG.read((char*)raw_grad_pairs.data(), sizeof(PersistencePair)*G);
        std::ifstream inGF(grad_filt_cache, std::ios::binary);
        size_t GF;
        inGF.read((char*)&GF, sizeof(GF));
        grad_filtration_values.resize(GF);
        inGF.read((char*)grad_filtration_values.data(), sizeof(int)*GF);
        std::cout << "Loaded " << raw_grad_pairs.size() << " gradient pairs from cache.\n";
    } else
    {
        Volume grad_vol = compute_gradient_volume(volume);
        raw_grad_pairs = calculate_persistence_pairs(grad_vol, grad_filtration_values, app_state.filtration_mode);
        std::filesystem::create_directories(cache_base);
        {
        std::ofstream outG(grad_pairs_cache, std::ios::binary);
        size_t G = raw_grad_pairs.size();
        outG.write((char*)&G, sizeof(G));
        outG.write((char*)raw_grad_pairs.data(), sizeof(PersistencePair)*G);
        }
        {
        std::ofstream outGF(grad_filt_cache, std::ios::binary);
        size_t GF = grad_filtration_values.size();
        outGF.write((char*)&GF, sizeof(GF));
        outGF.write((char*)grad_filtration_values.data(), sizeof(int)*GF);
        }
        std::cout << "Computed and cached " << raw_grad_pairs.size() << " gradient pairs.\n";
    }

    MergeTree merge_tree = build_merge_tree_with_tolerance(raw_pairs, 5);
    exportFilteredMergeTreeEdges(merge_tree, "merge_tree_edges_filtered.txt", 3, 10);
    std::vector<PersistencePair> defaultPairs = get_persistence_pairs_for_level(merge_tree, app_state.target_level);

    std::vector<PersistencePair> normalizedPairs;
    for (const PersistencePair &p : defaultPairs)
    {
        uint32_t normBirth = filtration_values[p.birth];
        uint32_t normDeath = filtration_values[p.death];
        normalizedPairs.push_back(PersistencePair(normBirth, normDeath));
    }

    std::ofstream outfile("volume_data/persistence_pairs.txt");
    if (outfile.is_open())
    {
        for (const auto &pair : raw_pairs)
        {
            outfile << filtration_values[pair.birth] << " " << filtration_values[pair.death] << "\n";
        }
        outfile.close();
        std::cout << "Persistence pairs saved to persistence_pairs.txt" << std::endl;
    } else
    {
        std::cerr << "Failed to open persistence_pairs.txt for writing!" << std::endl;
    }

    std::string outputFile = "persistence_diagram.png";
    std::string pythonCommand = "python scripts/persistence_diagram.py persistence_pairs.txt " + outputFile;
    int ret = system(pythonCommand.c_str());
    if (ret != 0) 
    {
        std::cerr << "Python script for persistence diagram failed with error code " << ret << std::endl;
    } else {
        std::cout << "Persistence diagram generated successfully." << std::endl;
    }

    std::vector<PersistencePair> displayPairs;
    displayPairs.reserve(raw_pairs.size());
    for (auto &p : raw_pairs)
    {
        // look up the true scalar
        uint32_t b = filtration_values[p.birth];
        uint32_t d = filtration_values[p.death];
        displayPairs.emplace_back(b, d);
    }

    std::vector<PersistencePair> gradDisplayPairs;
    for (auto &p : raw_grad_pairs)
    {
        gradDisplayPairs.emplace_back(grad_filtration_values[p.birth], grad_filtration_values[p.death]);
    }
    EventHandler eh;
    GPUContext gpu_context(app_state, volume);

    //gpu_context.wc.set_persistence_pairs(normalizedPairs, volume);
    gpu_context.wc.set_persistence_pairs(displayPairs, volume);
    //gpu_context.wc.set_raw_persistence_pairs(displayPairs);
    gpu_context.wc.set_gradient_persistence_pairs(gradDisplayPairs);

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
            merge_tree = build_merge_tree_with_tolerance(raw_pairs, 5);
            app_state.apply_filtration_mode = false;
        }

        if (app_state.apply_persistence_threshold)
        {
            std::vector<PersistencePair> currentFilteredPairs = threshold_cut(raw_pairs, app_state.persistence_threshold);
            std::cout << "Persistence threshold updated to " << app_state.persistence_threshold << ", filtered pairs: " << currentFilteredPairs.size() << std::endl;
            merge_tree = build_merge_tree_with_tolerance(currentFilteredPairs, 2);
            std::vector<PersistencePair> selectedPairs = get_persistence_pairs_for_level(merge_tree, app_state.target_level);
            gpu_context.wc.set_persistence_pairs(selectedPairs, volume);
            app_state.apply_persistence_threshold = false;
        }

        if (app_state.apply_target_level)
        {
            std::vector<PersistencePair> selectedPairs = get_persistence_pairs_for_level(merge_tree, app_state.target_level);
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

