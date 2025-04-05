#include "merge_tree.hpp"
#include <algorithm>
#include <iostream>

#include <optional>
#include <cmath>    

MergeTree::MergeTree() : root(nullptr) {}

MergeTree::~MergeTree() {
    for (auto& [id, node] : nodes) {
        delete node;
    }
}

void MergeTree::add_node(uint32_t id, uint32_t birth, uint32_t death) 
{
    if (nodes.find(id) != nodes.end()) 
    {
        std::cerr << "WARNING: Node ID=" << id << " already exists!" << std::endl;
    } else 
    {
        nodes[id] = new MergeTreeNode(id, birth, death);
        std::cout << "Added Node ID=" << id << ", Birth=" << birth << ", Death=" << death << std::endl;
        // update root: choose the node with the smallest birth
        if (!root || birth < root->birth) 
        {
            root = nodes[id];
        }
    }
}

MergeTreeNode* MergeTree::find(uint32_t id) 
{
    MergeTreeNode* node = nodes[id];
    if (node->parent == nullptr)
        return node;
    node->parent = find(node->parent->id);
    return node->parent;
}

void MergeTree::union_nodes(uint32_t idA, uint32_t idB) 
{
    if (nodes.find(idA) == nodes.end() || nodes.find(idB) == nodes.end()) 
    {
        std::cerr << "ERROR: Cannot union nodes " << idA << " and " << idB << " because one does not exist." << std::endl;
        return;
    }
    MergeTreeNode* repA = find(idA);
    MergeTreeNode* repB = find(idB);
    if (repA == repB)
        return;
    MergeTreeNode* parent = (repA->birth <= repB->birth) ? repA : repB;
    MergeTreeNode* child  = (parent == repA) ? repB : repA;
    child->parent = parent;
    child->depth = parent->depth + 1;
    parent->children.push_back(child);
    std::cout << "Union: Node " << parent->id << " (depth " << parent->depth << ") absorbs node " << child->id << " (depth " << child->depth << ")" << std::endl;
}

// directly attach death node to birth node without find/path compression
void MergeTree::chain_union(uint32_t birth_id, uint32_t death_id)
{
    if (nodes.find(birth_id) == nodes.end() || nodes.find(death_id) == nodes.end())
    {
        std::cerr << "Chain union error: one of the nodes does not exist." << std::endl;
        return;
    }
    MergeTreeNode* birthNode = nodes[birth_id];
    MergeTreeNode* deathNode = nodes[death_id];
    deathNode->parent = birthNode;
    deathNode->depth = birthNode->depth + 1;
    birthNode->children.push_back(deathNode);
    std::cout << "Chain Union: Node " << birthNode->id << " (depth " << birthNode->depth << ") absorbs node " << deathNode->id << " (depth " << deathNode->depth << ")" << std::endl;
}

MergeTreeNode* MergeTree::get_root() const
{
    return root;
}

const std::unordered_map<uint32_t, MergeTreeNode*>& MergeTree::get_all_nodes() const
{
    return nodes;
}

std::vector<uint32_t> MergeTree::find_nodes_by_depth(int targetDepth) const
{
    std::vector<uint32_t> result;
    for (const auto& [id, node] : nodes)
    {
        if (node->parent == nullptr && node->depth == targetDepth)
        {
            result.push_back(id);
        }
    }
    return result;
}

void MergeTree::set_target_level(int level)
{
    target_level = level;
}

void MergeTree::set_persistence_threshold(int threshold)
{
    persistence_threshold = threshold;
}

// find a key in compNodes within tolerance of deathVal
std::optional<uint32_t> findCloseKey(const std::unordered_map<uint32_t, uint32_t>& compNodes, uint32_t deathVal, uint32_t tol) 
{
    for (const auto &kv : compNodes) 
    {
        if (std::abs((int)kv.first - (int)deathVal) <= (int)tol) 
        {
            return kv.first;
        }
    }
    return std::nullopt;
}

MergeTree::MergeTree(MergeTree&& other) noexcept
    : nodes(std::move(other.nodes)),
      root(other.root),
      target_level(other.target_level),
      persistence_threshold(other.persistence_threshold)
{
    other.root = nullptr;
    other.nodes.clear();
}

MergeTree& MergeTree::operator=(MergeTree&& other) noexcept 
{
    if (this != &other) 
    {
        // delete current nodes to avoid memory leaks
        for (auto& [id, node] : nodes) 
        {
            delete node;
        }
        nodes = std::move(other.nodes);
        root = other.root;
        target_level = other.target_level;
        persistence_threshold = other.persistence_threshold;
        other.root = nullptr;
        other.nodes.clear();
    }
    return *this;
}

MergeTree build_merge_tree_with_tolerance(const std::vector<PersistencePair>& persistence_pairs, uint32_t tol) 
{
    MergeTree merge_tree;
    uint32_t uniqueId = 1;
    std::unordered_map<uint32_t, uint32_t> compNodes;
    
    auto pairs = persistence_pairs;
    std::sort(pairs.begin(), pairs.end(), [](const PersistencePair& a, const PersistencePair& b) {
         return a.birth < b.birth;
    });
    
    for (const auto &pair : pairs) 
    {
         // always create a new birth node
         uint32_t birth_node_id = uniqueId++;
         merge_tree.add_node(birth_node_id, pair.birth, pair.birth);
         
         // create a new death node for the merge event
         uint32_t death_node_id = uniqueId++;
         merge_tree.add_node(death_node_id, pair.birth, pair.death);
         
         // attach the death node as a child of the birth node
         merge_tree.chain_union(birth_node_id, death_node_id);
         
         // check for a near-equal death value within tolerance
         auto keyOpt = findCloseKey(compNodes, pair.death, tol);
         if(keyOpt.has_value()) 
         {
              uint32_t key = keyOpt.value();
              merge_tree.chain_union(compNodes[key], death_node_id);
              // remove old key and use the current death value
              compNodes.erase(key);
              compNodes[pair.death] = death_node_id;
         } else 
         {
              compNodes[pair.death] = death_node_id;
         }
    }
    return merge_tree;
}
