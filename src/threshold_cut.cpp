#include "threshold_cut.hpp"

std::vector<PersistencePair> threshold_cut(const std::vector<PersistencePair>& pairs, uint32_t threshold) 
{
    std::vector<PersistencePair> result;
    for (size_t i = 0; i < pairs.size(); ++i) {
        uint32_t p_i = (pairs[i].death > pairs[i].birth ? pairs[i].death - pairs[i].birth : 0);
        
        // only check if persistence is above threshold
        if (p_i >= threshold) {
            result.push_back(pairs[i]);
        }
    }
    return result;
}