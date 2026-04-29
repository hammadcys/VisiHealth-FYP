"use client";

import { useState, useEffect, useMemo } from "react";
import { motion } from "framer-motion";
import { ArrowLeft, History, Trash2, Download, Filter } from "lucide-react";
import Link from "next/link";
import { HistoryCard } from "@/components/history/HistoryCard";
import { SearchBar } from "@/components/history/SearchBar";
import { DetailModal } from "@/components/history/DetailModal";
import type { AnalysisHistory } from "@/types/api";
import toast from "react-hot-toast";

export default function HistoryPage() {
  const [history, setHistory] = useState<AnalysisHistory[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedItem, setSelectedItem] = useState<AnalysisHistory | null>(null);
  const [sortBy, setSortBy] = useState<"recent" | "confidence">("recent");

  // Load history from localStorage
  useEffect(() => {
    const loadHistory = () => {
      try {
        const savedHistory = localStorage.getItem("visihealth_history");
        if (savedHistory) {
          const parsed = JSON.parse(savedHistory);
          // One-time migration: strip heavy base64 fields from old entries
          // (imageUrl/attentionMap can be 200-800KB each and fill up the 5MB quota)
          const cleaned = parsed.map((entry: AnalysisHistory & { imageUrl?: string; attentionMap?: unknown }) => {
            // eslint-disable-next-line @typescript-eslint/no-unused-vars
            const { imageUrl, attentionMap, ...rest } = entry as { imageUrl?: string; attentionMap?: unknown } & AnalysisHistory;
            return rest;
          });
          setHistory(cleaned);
          // Re-save the cleaned entries so future loads are also lean
          try {
            localStorage.setItem("visihealth_history", JSON.stringify(cleaned));
          } catch {
            // If even saving cleaned data fails, wipe and start fresh
            localStorage.removeItem("visihealth_history");
            setHistory([]);
          }
        }
      } catch (error) {
        console.error("Failed to load history:", error);
        toast.error("Failed to load history");
      }
    };
    loadHistory();
  }, []);

  // Filter and sort history
  const filteredHistory = useMemo(() => {
    let filtered = history;

    // Search filter
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(
        (item) =>
          item.question.toLowerCase().includes(query) ||
          item.answer.toLowerCase().includes(query) ||
          item.topPredictions?.some((pred) =>
            pred.answer.toLowerCase().includes(query)
          )
      );
    }

    // Sort
    if (sortBy === "recent") {
      filtered = [...filtered].sort(
        (a, b) =>
          new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
      );
    } else {
      filtered = [...filtered].sort((a, b) => b.confidence - a.confidence);
    }

    return filtered;
  }, [history, searchQuery, sortBy]);

  const handleClearHistory = () => {
    if (confirm("Are you sure you want to clear all history?")) {
      localStorage.removeItem("visihealth_history");
      setHistory([]);
      toast.success("History cleared");
    }
  };

  const handleExportHistory = () => {
    try {
      const dataStr = JSON.stringify(history, null, 2);
      const dataUri =
        "data:application/json;charset=utf-8," + encodeURIComponent(dataStr);
      const exportFileDefaultName = `visihealth-history-${Date.now()}.json`;

      const linkElement = document.createElement("a");
      linkElement.setAttribute("href", dataUri);
      linkElement.setAttribute("download", exportFileDefaultName);
      linkElement.click();

      toast.success("History exported");
    } catch (error) {
      console.error("Export failed:", error);
      toast.error("Failed to export history");
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Header */}
        <div className="mb-8">
          <Link
            href="/"
            className="inline-flex items-center gap-2 text-blue-600 hover:text-blue-700 mb-4 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Home
          </Link>
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-center justify-between"
          >
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent flex items-center gap-3">
                <History className="w-10 h-10 text-blue-600" />
                Analysis History
              </h1>
              <p className="text-gray-600 mt-2">
                {history.length} {history.length === 1 ? "analysis" : "analyses"}{" "}
                saved
              </p>
            </div>
            <div className="flex gap-3">
              <button
                onClick={handleExportHistory}
                disabled={history.length === 0}
                className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <Download className="w-4 h-4" />
                Export
              </button>
              <button
                onClick={handleClearHistory}
                disabled={history.length === 0}
                className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <Trash2 className="w-4 h-4" />
                Clear All
              </button>
            </div>
          </motion.div>
        </div>

        {/* Search and Filters */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-xl shadow-md p-6 mb-8 border border-gray-100"
        >
          <div className="flex flex-col md:flex-row gap-4">
            <div className="flex-1">
              <SearchBar
                searchQuery={searchQuery}
                onSearchChange={setSearchQuery}
              />
            </div>
            <div className="flex items-center gap-2">
              <Filter className="w-5 h-5 text-gray-500" />
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value as "recent" | "confidence")}
                className="px-4 py-2 border-2 border-gray-200 rounded-lg focus:border-blue-500 focus:ring-2 focus:ring-blue-200 outline-none transition-all"
              >
                <option value="recent">Most Recent</option>
                <option value="confidence">Highest Confidence</option>
              </select>
            </div>
          </div>
        </motion.div>

        {/* History Grid */}
        {filteredHistory.length === 0 ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="bg-white rounded-xl shadow-md p-12 text-center border border-gray-100"
          >
            <div className="w-20 h-20 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <History className="w-10 h-10 text-gray-400" />
            </div>
            <h3 className="text-xl font-semibold text-gray-700 mb-2">
              {searchQuery
                ? "No results found"
                : "No analysis history yet"}
            </h3>
            <p className="text-gray-500 mb-6">
              {searchQuery
                ? "Try a different search term"
                : "Start by analyzing your first medical image"}
            </p>
            {!searchQuery && (
              <Link
                href="/analyze"
                className="inline-block px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                Analyze Now
              </Link>
            )}
          </motion.div>
        ) : (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
          >
            {filteredHistory.map((item, index) => (
              <motion.div
                key={item.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
              >
                <HistoryCard item={item} onClick={() => setSelectedItem(item)} />
              </motion.div>
            ))}
          </motion.div>
        )}
      </div>

      {/* Detail Modal */}
      <DetailModal item={selectedItem} onClose={() => setSelectedItem(null)} />
    </div>
  );
}
