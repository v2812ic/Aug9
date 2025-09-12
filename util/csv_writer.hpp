#pragma once
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <mutex>
#include <unordered_set>

namespace io {

class CsvWriter {
public:
  static void append_row(const std::string& path,
                         const std::vector<std::string>& header,
                         const std::vector<double>& row) {
    std::scoped_lock<std::mutex> lk(mutex_);
    ensure_parent_dir(path);
    const bool file_exists = std::filesystem::exists(path);

    std::ofstream ofs(path, std::ios::app);
    if (!ofs.is_open()) throw std::runtime_error("Cannot open " + path);

    if (!file_exists) {
      // header
      for (size_t i = 0; i < header.size(); ++i) {
        ofs << header[i] << (i + 1 < header.size() ? "," : "\n");
      }
    }
    // row
    for (size_t i = 0; i < row.size(); ++i) {
      ofs << row[i] << (i + 1 < row.size() ? "," : "\n");
    }
  }

  static void write_text(const std::string& path, const std::string& text) {
    std::scoped_lock<std::mutex> lk(mutex_);
    ensure_parent_dir(path);
    std::ofstream ofs(path, std::ios::out);
    if (!ofs.is_open()) throw std::runtime_error("Cannot open " + path);
    ofs << text;
  }

private:
  static void ensure_parent_dir(const std::string& path) {
    std::filesystem::path p(path);
    if (p.has_parent_path()) {
      std::filesystem::create_directories(p.parent_path());
    }
  }
  static inline std::mutex mutex_;
};

}