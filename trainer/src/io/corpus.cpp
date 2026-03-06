// ============================================================
// FILE: trainer/src/io/corpus.cpp
// ============================================================
#include "io/corpus.h"
#include "io/tokenizer.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <functional>

namespace io {

void corpus_write(const std::string& path, const std::vector<uint32_t>& tokens) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { perror(("Cannot write " + path).c_str()); exit(1); }
    uint32_t magic   = CORPUS_MAGIC;
    uint32_t version = CORPUS_VERSION;
    uint64_t count   = tokens.size();
    fwrite(&magic,   4, 1, f);
    fwrite(&version, 4, 1, f);
    fwrite(&count,   8, 1, f);
    fwrite(tokens.data(), sizeof(uint32_t), tokens.size(), f);
    fclose(f);
    printf("Wrote %llu tokens to %s\n", (unsigned long long)count, path.c_str());
}

void CorpusLoader::open(const std::string& path) {
    fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) { perror(("Cannot open corpus " + path).c_str()); exit(1); }
    struct stat st; fstat(fd, &st);
    mmap_size = st.st_size;
    mmap_ptr  = mmap(nullptr, mmap_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mmap_ptr == MAP_FAILED) { perror("mmap corpus"); exit(1); }

    const char* base = (const char*)mmap_ptr;
    uint32_t magic, version;
    memcpy(&magic,   base + 0, 4);
    memcpy(&version, base + 4, 4);
    memcpy(&num_tokens, base + 8, 8);
    if (magic != CORPUS_MAGIC) {
        fprintf(stderr, "Bad corpus magic: %08x\n", magic); exit(1);
    }
    token_ptr = (const uint32_t*)(base + 16);  // 4+4+8 = 16 bytes header
}

void CorpusLoader::close() {
    if (mmap_ptr && mmap_ptr != MAP_FAILED) { munmap(mmap_ptr, mmap_size); mmap_ptr = nullptr; }
    if (fd >= 0) { ::close(fd); fd = -1; }
    num_tokens = 0; token_ptr = nullptr;
}

bool CorpusLoader::next_batch(uint32_t* input, uint32_t* target,
                               int batch, int seq, uint64_t& offset) const {
    // Each sample needs (seq+1) tokens (for input + target shift).
    // We pack batch*seq tokens into input and batch*seq into target.
    uint64_t needed = (uint64_t)batch * seq + 1;
    if (offset + needed > num_tokens) return false;

    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq; ++s) {
            uint64_t idx = offset + (uint64_t)b * seq + s;
            input [b * seq + s] = token_ptr[idx];
            target[b * seq + s] = token_ptr[idx + 1];
        }
    }
    offset += (uint64_t)batch * seq;
    return true;
}

// ---- tokenize subcommand implementation ----
void tokenize_dir(const std::string& model_dir,
                  const std::string& in_path,
                  const std::string& out_path) {
    Tokenizer tok;
    tok.load(model_dir);

    std::vector<uint32_t> all_tokens;

    // Check if a file looks like text (not binary).
    // Sample the first 8KB and reject if >5% null or non-printable bytes.
    auto is_text_file = [](const std::string& fpath) -> bool {
        FILE* f = fopen(fpath.c_str(), "rb");
        if (!f) return false;
        unsigned char buf[8192];
        size_t n = fread(buf, 1, sizeof(buf), f);
        fclose(f);
        if (n == 0) return false;
        size_t bad = 0;
        for (size_t i = 0; i < n; ++i) {
            // null bytes and non-printable control chars (except tab, newline, CR)
            if (buf[i] == 0 ||
                (buf[i] < 8) ||
                (buf[i] > 13 && buf[i] < 32 && buf[i] != 27)) {
                ++bad;
            }
        }
        return (bad * 100 / n) < 5;
    };

    // Recursive directory walk using a stack.
    auto process_file = [&](const std::string& fpath) {
        std::ifstream f(fpath);
        if (!f) { fprintf(stderr, "  Cannot read %s, skipping.\n", fpath.c_str()); return; }
        std::ostringstream ss; ss << f.rdbuf();
        std::string text = ss.str();
        if (text.empty()) return;
        auto ids = tok.encode(text);
        ids.push_back(tok.eos_id);
        all_tokens.insert(all_tokens.end(), ids.begin(), ids.end());
        printf("  %s → %zu tokens\n", fpath.c_str(), ids.size());
    };

    std::function<void(const std::string&)> walk = [&](const std::string& dir) {
        DIR* d = opendir(dir.c_str());
        if (!d) { perror(("Cannot open dir " + dir).c_str()); return; }
        struct dirent* ent;
        std::vector<std::string> subdirs;
        std::vector<std::string> files;
        while ((ent = readdir(d)) != nullptr) {
            std::string name = ent->d_name;
            if (name == "." || name == "..") continue;
            std::string full = dir + "/" + name;
            if (ent->d_type == DT_DIR) {
                subdirs.push_back(full);
            } else if (ent->d_type == DT_REG || ent->d_type == DT_UNKNOWN) {
                files.push_back(full);
            }
        }
        closedir(d);
        std::sort(files.begin(), files.end());
        std::sort(subdirs.begin(), subdirs.end());
        for (auto& fp : files) {
            if (is_text_file(fp)) process_file(fp);
            else printf("  skipping binary: %s\n", fp.c_str());
        }
        for (auto& sd : subdirs) walk(sd);
    };

    // Handle both single file and directory input
    struct stat st;
    stat(in_path.c_str(), &st);
    if (S_ISDIR(st.st_mode)) {
        walk(in_path);
    } else {
        if (is_text_file(in_path)) process_file(in_path);
        else fprintf(stderr, "Input file appears to be binary, skipping.\n");
    }

    printf("Total tokens: %zu\n", all_tokens.size());
    corpus_write(out_path, all_tokens);
}

} // namespace io