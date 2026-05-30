#include <iostream>
#include <vector>
#include <deque>
#include <algorithm>
#include <cstdint>
#include <queue>
#include <string>
#include "robin_hood.h"

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <cstdio>
#endif

using namespace std;

int WINDOW = 5; // Default window size
const size_t MEMORY_LIMIT_PAIRS = 250000000; // Dump to disk when we reach 250M unique pairs (~3.2GB map + ~2.0GB temp sort vector)

inline uint32_t pack(uint16_t a, uint16_t b) {
    if (a > b) swap(a, b);
    return ((uint32_t)a << 16) | b;
}

inline pair<uint16_t,uint16_t> unpack(uint32_t p) {
    return {p >> 16, p & 0xFFFF};
}

// Write the current in-memory map to a sorted chunk file on disk
void write_chunk(const robin_hood::unordered_flat_map<uint32_t, uint32_t>& counts, int chunk_idx, vector<string>& temp_files) {
    string filename = "temp_chunk_" + to_string(chunk_idx) + ".bin";
    temp_files.push_back(filename);

    cerr << "Dumping chunk " << chunk_idx << " to " << filename << " (unique pairs: " << counts.size() << ")..." << endl;

    // 1. Copy map contents to a vector for sorting (manual copy for old compiler compatibility)
    vector<pair<uint32_t, uint32_t>> vec;
    vec.reserve(counts.size());
    for (const auto& p : counts) {
        vec.push_back({p.first, p.second});
    }

    // 2. Sort by key (which naturally sorts by token i, then token j)
    sort(vec.begin(), vec.end(), [](const pair<uint32_t, uint32_t>& a, const pair<uint32_t, uint32_t>& b) {
        return a.first < b.first;
    });

    // 3. Write to binary file
    FILE* f = fopen(filename.c_str(), "wb");
    if (!f) {
        cerr << "ERROR: cannot open temp file " << filename << endl;
        exit(1);
    }
    for (const auto& p : vec) {
        uint16_t i = p.first >> 16;
        uint16_t j = p.first & 0xFFFF;
        uint32_t c = p.second;
        fwrite(&i, 2, 1, f);
        fwrite(&j, 2, 1, f);
        fwrite(&c, 4, 1, f);
    }
    fclose(f);
}

// Structure for the priority queue item in k-way merge
struct MergeItem {
    uint32_t key;
    uint32_t count;
    int file_idx;

    // Min-heap: smallest key goes first
    bool operator>(const MergeItem& other) const {
        return key > other.key;
    }
};

// Stream-merge all sorted chunk files from disk to the final output file
void merge_chunks(const vector<string>& temp_files, const char* outfile_path) {
    cerr << "Merging " << temp_files.size() << " chunk files into " << outfile_path << "..." << endl;

    vector<FILE*> files;
    priority_queue<MergeItem, vector<MergeItem>, greater<MergeItem>> pq;

    // Open all chunk files and push their first records into the priority queue
    for (int i = 0; i < (int)temp_files.size(); ++i) {
        FILE* f = fopen(temp_files[i].c_str(), "rb");
        if (!f) {
            cerr << "ERROR: cannot open chunk file " << temp_files[i] << endl;
            exit(1);
        }
        files.push_back(f);

        uint16_t ti, tj;
        uint32_t tc;
        if (fread(&ti, 2, 1, f) && fread(&tj, 2, 1, f) && fread(&tc, 4, 1, f)) {
            uint32_t key = ((uint32_t)ti << 16) | tj;
            pq.push({key, tc, i});
        }
    }

    FILE* out = fopen(outfile_path, "wb");
    if (!out) {
        cerr << "ERROR: cannot open output file " << outfile_path << endl;
        exit(1);
    }

    uint32_t current_key = 0xFFFFFFFF;
    uint64_t current_count = 0;
    uint64_t written = 0;

    while (!pq.empty()) {
        MergeItem top = pq.top();
        pq.pop();

        if (top.key == current_key) {
            current_count += top.count;
        } else {
            // Write previous aggregated pair if valid
            if (current_key != 0xFFFFFFFF) {
                uint16_t i = current_key >> 16;
                uint16_t j = current_key & 0xFFFF;
                uint32_t c = (current_count > 0xFFFFFFFF) ? 0xFFFFFFFF : (uint32_t)current_count;
                fwrite(&i, 2, 1, out);
                fwrite(&j, 2, 1, out);
                fwrite(&c, 4, 1, out);
                written++;
            }
            current_key = top.key;
            current_count = top.count;
        }

        // Read next record from the file that just yielded the top item
        int idx = top.file_idx;
        FILE* f = files[idx];
        uint16_t ti, tj;
        uint32_t tc;
        if (fread(&ti, 2, 1, f) && fread(&tj, 2, 1, f) && fread(&tc, 4, 1, f)) {
            uint32_t key = ((uint32_t)ti << 16) | tj;
            pq.push({key, tc, idx});
        }
    }

    // Write the last aggregated pair
    if (current_key != 0xFFFFFFFF) {
        uint16_t i = current_key >> 16;
        uint16_t j = current_key & 0xFFFF;
        uint32_t c = (current_count > 0xFFFFFFFF) ? 0xFFFFFFFF : (uint32_t)current_count;
        fwrite(&i, 2, 1, out);
        fwrite(&j, 2, 1, out);
        fwrite(&c, 4, 1, out);
        written++;
    }

    fclose(out);

    // Close and remove all temporary chunk files
    for (int i = 0; i < (int)files.size(); ++i) {
        fclose(files[i]);
        remove(temp_files[i].c_str());
    }

    cerr << "Saved " << written << " unique pairs to " << outfile_path << endl;
}

// 2-way stream merge of two sorted co-occurrence binary files
void stream_merge(const char* f1_path, const char* f2_path, const char* out_path) {
    cerr << "Merging " << f1_path << " and " << f2_path << " into " << out_path << "..." << endl;
    FILE* f1 = fopen(f1_path, "rb");
    FILE* f2 = fopen(f2_path, "rb");
    FILE* out = fopen(out_path, "wb");

    if (!out) {
        cerr << "ERROR: cannot open output file " << out_path << endl;
        if (f1) fclose(f1);
        if (f2) fclose(f2);
        exit(1);
    }

    if (!f1 && !f2) {
        fclose(out);
        return;
    }

    if (!f1) {
        // Copy f2 to out
        uint8_t buffer[65536];
        size_t bytes;
        while ((bytes = fread(buffer, 1, sizeof(buffer), f2)) > 0) {
            fwrite(buffer, 1, bytes, out);
        }
        fclose(f2);
        fclose(out);
        return;
    }

    if (!f2) {
        // Copy f1 to out
        uint8_t buffer[65536];
        size_t bytes;
        while ((bytes = fread(buffer, 1, sizeof(buffer), f1)) > 0) {
            fwrite(buffer, 1, bytes, out);
        }
        fclose(f1);
        fclose(out);
        return;
    }

    struct Record {
        uint16_t i;
        uint16_t j;
        uint32_t count;
        uint32_t key() const {
            return ((uint32_t)i << 16) | j;
        }
    };

    Record r1, r2;
    bool has_r1 = fread(&r1.i, 2, 1, f1) && fread(&r1.j, 2, 1, f1) && fread(&r1.count, 4, 1, f1);
    bool has_r2 = fread(&r2.i, 2, 1, f2) && fread(&r2.j, 2, 1, f2) && fread(&r2.count, 4, 1, f2);

    uint64_t written = 0;

    while (has_r1 && has_r2) {
        uint32_t k1 = r1.key();
        uint32_t k2 = r2.key();

        if (k1 < k2) {
            fwrite(&r1.i, 2, 1, out);
            fwrite(&r1.j, 2, 1, out);
            fwrite(&r1.count, 4, 1, out);
            written++;
            has_r1 = fread(&r1.i, 2, 1, f1) && fread(&r1.j, 2, 1, f1) && fread(&r1.count, 4, 1, f1);
        } else if (k2 < k1) {
            fwrite(&r2.i, 2, 1, out);
            fwrite(&r2.j, 2, 1, out);
            fwrite(&r2.count, 4, 1, out);
            written++;
            has_r2 = fread(&r2.i, 2, 1, f2) && fread(&r2.j, 2, 1, f2) && fread(&r2.count, 4, 1, f2);
        } else {
            // equal keys, merge counts
            uint64_t merged_count = (uint64_t)r1.count + r2.count;
            uint32_t c = (merged_count > 0xFFFFFFFF) ? 0xFFFFFFFF : (uint32_t)merged_count;
            fwrite(&r1.i, 2, 1, out);
            fwrite(&r1.j, 2, 1, out);
            fwrite(&c, 4, 1, out);
            written++;
            has_r1 = fread(&r1.i, 2, 1, f1) && fread(&r1.j, 2, 1, f1) && fread(&r1.count, 4, 1, f1);
            has_r2 = fread(&r2.i, 2, 1, f2) && fread(&r2.j, 2, 1, f2) && fread(&r2.count, 4, 1, f2);
        }
    }

    while (has_r1) {
        fwrite(&r1.i, 2, 1, out);
        fwrite(&r1.j, 2, 1, out);
        fwrite(&r1.count, 4, 1, out);
        written++;
        has_r1 = fread(&r1.i, 2, 1, f1) && fread(&r1.j, 2, 1, f1) && fread(&r1.count, 4, 1, f1);
    }

    while (has_r2) {
        fwrite(&r2.i, 2, 1, out);
        fwrite(&r2.j, 2, 1, out);
        fwrite(&r2.count, 4, 1, out);
        written++;
        has_r2 = fread(&r2.i, 2, 1, f2) && fread(&r2.j, 2, 1, f2) && fread(&r2.count, 4, 1, f2);
    }

    fclose(f1);
    fclose(f2);
    fclose(out);
    cerr << "Merge completed. Saved " << written << " unique pairs to " << out_path << endl;
}

int main(int argc, char* argv[]) {
    if (argc >= 5 && string(argv[1]) == "--merge") {
        const char* f1_path = argv[2];
        const char* f2_path = argv[3];
        const char* out_path = argv[4];
        stream_merge(f1_path, f2_path, out_path);
        return 0;
    }

    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <input_tokens.bin> <output_counts.bin> [window_size]" << endl;
        cerr << "   or: " << argv[0] << " --merge <file1.bin> <file2.bin> <output.bin>" << endl;
        return 1;
    }

    if (argc >= 4) {
        try {
            WINDOW = stoi(argv[3]);
            if (WINDOW <= 0) {
                cerr << "ERROR: window size must be > 0" << endl;
                return 1;
            }
        } catch (...) {
            cerr << "WARNING: Invalid window size argument, defaulting to " << WINDOW << endl;
        }
    }

    const char* infile_path = argv[1];
    const char* outfile_path = argv[2];

    FILE* infile = fopen(infile_path, "rb");
    if (!infile) {
        cerr << "ERROR: cannot open input file " << infile_path << endl;
        return 1;
    }

    // Get input file size to calculate total tokens for progress bar
    fseek(infile, 0, SEEK_END);
    long long file_size = ftell(infile);
    fseek(infile, 0, SEEK_SET);
    uint64_t total_tokens = file_size / 2;
    cerr << "Total tokens to process: " << total_tokens << endl;

    robin_hood::unordered_flat_map<uint32_t, uint32_t> counts;
    counts.reserve(10000000);

    deque<uint16_t> buf;
    uint64_t tokens = 0;
    int chunk_idx = 0;
    vector<string> temp_files;

    const size_t BUFFER_SIZE = 1000000; // 1M tokens (2MB buffer)
    vector<uint16_t> read_buf(BUFFER_SIZE);

    uint64_t next_progress_check = 1000000;

    // ── Pass 1: sliding window counting with periodic memory dumping ──
    while (true) {
        size_t n_read = fread(read_buf.data(), 2, BUFFER_SIZE, infile);
        if (n_read <= 0) {
            break;
        }
        for (size_t k = 0; k < n_read; ++k) {
            uint16_t t = read_buf[k];
            buf.push_back(t);

            if ((int)buf.size() > WINDOW * 2 + 1)
                buf.pop_front();

            int center = (int)buf.size() / 2;
            uint16_t w = buf[center];

            for (int i = 0; i < (int)buf.size(); i++) {
                if (i == center) continue;
                uint16_t c = buf[i];
                if (c == w) continue;
                counts[pack(w, c)]++;
            }

            tokens++;
            
            // Periodically check if hash map is too large and dump it
            if (counts.size() >= MEMORY_LIMIT_PAIRS) {
                write_chunk(counts, chunk_idx++, temp_files);
                counts.clear(); // Free memory
                counts.reserve(10000000);
            }

            if (tokens >= next_progress_check) {
                float progress = (total_tokens > 0) ? (float)tokens / total_tokens : 0.0f;
                int bar_width = 30;
                int pos = bar_width * progress;
                
                cerr << "\rCounting tokens: [";
                for (int i = 0; i < bar_width; ++i) {
                    if (i < pos) cerr << "=";
                    else if (i == pos) cerr << ">";
                    else cerr << " ";
                }
                cerr << "] " << int(progress * 100.0) << "% (" 
                     << tokens / 1000000 << "M/" << total_tokens / 1000000 << "M tokens | unique active: " 
                     << counts.size() << ")";
                cerr.flush();
                next_progress_check += 5000000; // Update every 5M tokens
            }
        }
    }
    fclose(infile);

    // Print final completed bar
    cerr << "\rCounting tokens: [==============================] 100% (" 
         << tokens / 1000000 << "M/" << total_tokens / 1000000 << "M tokens)" << endl;
    cerr << "Counting phase finished. Total tokens: " << tokens << endl;

    // If we have existing temp files, we must write the final memory slice as a chunk and merge them all
    if (!temp_files.empty()) {
        if (!counts.empty()) {
            write_chunk(counts, chunk_idx++, temp_files);
            counts.clear();
        }
        merge_chunks(temp_files, outfile_path);
    } else {
        // If we never hit the memory limit, just write counts directly (keeps it extremely fast)
        cerr << "No memory limit reached. Writing output directly..." << endl;
        FILE* f = fopen(outfile_path, "wb");
        if (!f) {
            cerr << "ERROR: cannot open output file " << outfile_path << endl;
            return 1;
        }

        // Sort counts before writing to maintain sorted outputs (manual copy for old compiler compatibility)
        vector<pair<uint32_t, uint32_t>> vec;
        vec.reserve(counts.size());
        for (const auto& p : counts) {
            vec.push_back({p.first, p.second});
        }
        sort(vec.begin(), vec.end(), [](const pair<uint32_t, uint32_t>& a, const pair<uint32_t, uint32_t>& b) {
            return a.first < b.first;
        });

        uint64_t written = 0;
        for (const auto& p : vec) {
            uint16_t i = p.first >> 16;
            uint16_t j = p.first & 0xFFFF;
            uint32_t c = p.second;
            fwrite(&i, 2, 1, f);
            fwrite(&j, 2, 1, f);
            fwrite(&c, 4, 1, f);
            written++;
        }
        fclose(f);
        cerr << "Saved " << written << " unique pairs to " << outfile_path << endl;
    }

    return 0;
}