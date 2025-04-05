#pragma once
#include <vector>
#include <unordered_map>
#include <cstdint>
#include "persistence.hpp"

struct MergeTreeNode 
{
    uint32_t id;
    uint32_t birth;
    uint32_t death;
    int depth;
    MergeTreeNode* parent;
    std::vector<MergeTreeNode*> children;

    MergeTreeNode(uint32_t id, uint32_t birth, uint32_t death) : id(id), birth(birth), death(death), depth(0), parent(nullptr) {}
};

class MergeTree 
{
public:
    MergeTree();
    ~MergeTree();

    void add_node(uint32_t id, uint32_t birth, uint32_t death);

    // standard find with path compression
    MergeTreeNode* find(uint32_t id);
    
    // standard union (which flattens the tree)
    void union_nodes(uint32_t id1, uint32_t id2);

    // attaches death node directly to birth node without path compression
    void chain_union(uint32_t birth_id, uint32_t death_id);

    MergeTreeNode* get_root() const;

    const std::unordered_map<uint32_t, MergeTreeNode*>& get_all_nodes() const;

    std::vector<uint32_t> find_nodes_by_depth(int targetDepth) const;

    void set_target_level(int level);
    void set_persistence_threshold(int threshold);

    MergeTree(MergeTree&& other) noexcept;
    MergeTree& operator=(MergeTree&& other) noexcept;

private:
    std::unordered_map<uint32_t, MergeTreeNode*> nodes;
    MergeTreeNode* root;
    int target_level = 0;
    int persistence_threshold = 0;

    void update_visualization();
};

MergeTree build_merge_tree_with_tolerance(const std::vector<PersistencePair>& persistence_pairs, uint32_t tol);
