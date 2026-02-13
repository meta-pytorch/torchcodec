// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <list>
#include <stdexcept>
#include <unordered_map>

namespace facebook::torchcodec {

// Eviction Policy Interface (concept):
// Each policy must provide:
//   - void onInsert(CacheIterator it)      // Called when entry is added
//   - void onRemove(CacheIterator it)      // Called when entry is removed
//   - CacheIterator selectForEviction()    // Returns iterator to evict
//   - bool empty() const                   // Returns true if nothing to evict

// LRU (Least Recently Used) eviction policy.
// Tracks cache entries in access order using a doubly-linked list.
// All operations are O(1).
template <typename CacheIterator>
class LRUEvictionPolicy {
 public:
  // Called when a decoder is returned to cache.
  // New entries go to front (most recently used).
  void onInsert(CacheIterator it) {
    lruList_.push_front(it);
    iteratorMap_[iteratorToKey(it)] = lruList_.begin();
  }

  // Called when a decoder is taken from cache or evicted.
  void onRemove(CacheIterator it) {
    auto mapIt = iteratorMap_.find(iteratorToKey(it));
    if (mapIt != iteratorMap_.end()) {
      lruList_.erase(mapIt->second);
      iteratorMap_.erase(mapIt);
    }
  }

  // Returns iterator to least recently used entry (back of list).
  CacheIterator selectForEviction() {
    return lruList_.back();
  }

  bool empty() const {
    return lruList_.empty();
  }

 private:
  // Use address of the pair in the multimap as key.
  // Multimap iterators are stable, so this is safe.
  static const void* iteratorToKey(CacheIterator it) {
    return &(*it);
  }

  // Front = most recently used, Back = least recently used
  std::list<CacheIterator> lruList_;
  // Map from cache entry address to position in LRU list
  std::unordered_map<const void*, typename std::list<CacheIterator>::iterator>
      iteratorMap_;
};

// No-eviction policy: rejects new entries when cache is full.
// This matches the original behavior before LRU was added.
template <typename CacheIterator>
class NoEvictionPolicy {
 public:
  void onInsert(CacheIterator) {}

  void onRemove(CacheIterator) {}

  CacheIterator selectForEviction() {
    // This should never be called since empty() always returns true
    throw std::logic_error(
        "NoEvictionPolicy::selectForEviction should not be called");
  }

  // Always returns true to signal that eviction is not available
  bool empty() const {
    return true;
  }
};

// FIFO (First In, First Out) eviction policy.
// Evicts the oldest entry regardless of access pattern.
// All operations are O(1).
template <typename CacheIterator>
class FIFOEvictionPolicy {
 public:
  void onInsert(CacheIterator it) {
    // New entries go to back (newest)
    fifoList_.push_back(it);
    iteratorMap_[iteratorToKey(it)] = std::prev(fifoList_.end());
  }

  void onRemove(CacheIterator it) {
    auto mapIt = iteratorMap_.find(iteratorToKey(it));
    if (mapIt != iteratorMap_.end()) {
      fifoList_.erase(mapIt->second);
      iteratorMap_.erase(mapIt);
    }
  }

  // Returns iterator to oldest entry (front of list).
  CacheIterator selectForEviction() {
    return fifoList_.front();
  }

  bool empty() const {
    return fifoList_.empty();
  }

 private:
  static const void* iteratorToKey(CacheIterator it) {
    return &(*it);
  }

  // Front = oldest, Back = newest
  std::list<CacheIterator> fifoList_;
  std::unordered_map<const void*, typename std::list<CacheIterator>::iterator>
      iteratorMap_;
};

} // namespace facebook::torchcodec
