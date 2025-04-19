#include "threshold_cut.hpp"

#include <algorithm>

std::vector<PersistencePair> threshold_cut(const std::vector<PersistencePair>& pairs, uint32_t threshold) 
{
    // The pairs vector must be sorted in ascending order by persistence.
    auto comp = [threshold](const PersistencePair& p, uint32_t thresh) -> bool
    {
        uint32_t pers = (p.death > p.birth ? p.death - p.birth : 0);
        return pers < thresh;
    };
    auto it = std::lower_bound(pairs.begin(), pairs.end(), threshold, comp);
    std::vector<PersistencePair> result(it, pairs.end());
    return result;
}

// keep only those pairs whose distance from the diagonal is >= minDistance
std::vector<PersistencePair> diagonal_distance_cut(const std::vector<PersistencePair>& pairs, float minDistance)
{
    std::vector<PersistencePair> result;
    float sqrt2 = std::sqrt(2.0f);
    for (const auto &p : pairs)
    {
        float pers = (p.death > p.birth) ? (p.death - p.birth) : 0.0f;

        // euclidean distance from the line death=birth is pers / sqrt(2)
        float distFromDiagonal = pers / sqrt2;

        // keep only pairs with distance >= minDistance
        if (distFromDiagonal >= minDistance)
        {
            result.push_back(p);
        }
    }
    return result;
}

std::vector<PersistencePair> filter_non_egenerate(const std::vector<PersistencePair>& pairs, uint32_t minPersistence)
{
    std::vector<PersistencePair> filtered;
    for (const auto& p : pairs)
    {
        if (p.death > p.birth + minPersistence)
        {
            filtered.push_back(p);
        }
    }
    return filtered;
}
