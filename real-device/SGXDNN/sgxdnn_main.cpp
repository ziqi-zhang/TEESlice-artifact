#define USE_EIGEN_TENSOR

#ifndef USE_SGX
#define EIGEN_USE_THREADS
#include <malloc.h>
#else
#include "Enclave.h"
#include "sgx_tseal.h"
#include "sgx_trts.h"
#include "sgx_thread.h"
#endif

#include "sgxdnn_main.hpp"
#include "randpool.hpp"
#include "utils.hpp"

#include "common_with_enclaves.h"

#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <chrono>
#include <string>
#include <cstring>
#include <cmath>
#include <deque>
#include <unordered_map>
#include <cstdlib>
#include <mutex>
#include <stack>
#include <time.h>
#include "Crypto.h"
#include <omp.h>
#include "../App/common_utils.cpp"


using namespace std;

using std::shared_ptr;
using std::make_shared;
using std::unordered_map;
using std::string;
using defer = shared_ptr<void>;


//using namespace SGXDNN;

int p_int = PrimeLimit;
float p = (float) p_int;
float mid = (float) (p_int / 2);

// some vectorized constants
__m256 p8f = _mm256_set1_ps(p);
__m256 p28f = _mm256_set1_ps(p * 2);
__m256 mid8f = _mm256_set1_ps(mid);
__m256 pmid8f = _mm256_set1_ps(p + mid);
__m256 negmid8f = _mm256_set1_ps(-mid - 1);
__m256 zero8f = _mm256_set1_ps((float)(0));
__m256 inv_shift8f = _mm256_set1_ps((float)(1.0/256));
__m256 six8f = _mm256_set1_ps((float) 6 * 256 * 256);

inline void MoveDown(float* input, float* out, int num_elements) {
	for(size_t i = 0; i < num_elements; i += 8) {
			const __m256 inp8f = _mm256_load_ps( &input[i] );             // blinded input

			const __m256 if_geq  = _mm256_cmp_ps(inp8f, mid8f, 0x0d);    // unblinded >= mid
			// const __m256 if_lt   = _mm256_cmp_ps(inp8f, negmid8f, 0x01);  // unblinded < -mid
			const __m256 then8f  = _mm256_sub_ps(inp8f, p8f);            // unblinded - p
			// const __m256 elif8f  = _mm256_add_ps(inp8f, p8f);            // unblinded + p
			const __m256 res8f = _mm256_blendv_ps(
                                        inp8f,
										then8f,
										if_geq);

			_mm256_stream_ps(&out[i], res8f);
    }
}


void ModP(MapMatRowMajor& m) {
    DtypeForCpuOp PLimit = static_cast<DtypeForCpuOp>(PrimeLimit);
    DtypeForCpuOp invPLimit = static_cast<DtypeForCpuOp>(1) / PrimeLimit;
    m.array() = m.array() - (m * invPLimit).array() * PLimit;
}

void ModP(EigenTensor& m) {
    DtypeForCpuOp PLimit = static_cast<DtypeForCpuOp>(PrimeLimit);
    DtypeForCpuOp invPLimit = static_cast<double>(1) / PrimeLimit;
    m -= (m * invPLimit).floor() * PLimit;
    // m = (m > m.constant((float) HalfPrime)).select(m - (float) HalfPrime, m);
}

void ModP(MapEigenTensor& m) {
    DtypeForCpuOp PLimit = static_cast<DtypeForCpuOp>(PrimeLimit);
    DtypeForCpuOp invPLimit = static_cast<double>(1) / PrimeLimit;
    m -= (m * invPLimit).floor() * PLimit;
    // m = (m > m.constant((float) HalfPrime)).select(m - (float) HalfPrime, m);
}

// #define PRINT_CHUNK_INFO 
// #define PRINT_CONV_OUTPUT_SAVE_CHUNK_INFO
// #define PRINT_CONV_INPUT_LOAD_CHUNK_INFO
// #define PRINT_CONV_IM2COL_CONSTRUCT_INFO
// #define PRINT_CONV_INIT_INFO
// #define PRINT_RUN_TIME_INFO

class ChunkPool {
public:
    ChunkPool(int size_pool_, int num_byte_chunk_) :
        size_pool(size_pool_),
        num_byte_chunk(num_byte_chunk_)
    {
        for (int i = 0; i < size_pool; i++) {
            void* enc_chunk = (void*)memalign(64, num_byte_chunk);
            chunks.push_back(enc_chunk);
            chunk_ids.push(i);
        }
        #ifdef PRINT_CHUNK_INFO
            printf("Pool size %d, num_byte_chunk %d\n", size_pool, num_byte_chunk);
        #endif
    }

    int get_chunk_id() {
        std::unique_lock<std::mutex> lock(stack_mutex);
        if (chunk_ids.empty()) {
            printf("Running out of chunks\n");
            throw std::invalid_argument("Running out of chunks");
        }
        int res;
        res = chunk_ids.top();
        chunk_ids.pop();
        return res;
    }

    void return_chunk_id(int id) {
        std::unique_lock<std::mutex> lock(stack_mutex);
        chunk_ids.push(id);
    }

    std::vector<void*> chunks;

private:
    int size_pool;
    int num_byte_chunk;
    std::mutex stack_mutex;
    std::stack<int> chunk_ids;
};

class StoreChunkPool {
public:
    static shared_ptr<ChunkPool> GetChunkPool() {
        static StoreChunkPool instance;
        return instance.chunk_pool;
    }
    StoreChunkPool(StoreChunkPool const&) = delete;
    void operator=(StoreChunkPool const&) = delete;

private:
    StoreChunkPool() {
        chunk_pool = make_shared<ChunkPool>(THREAD_POOL_SIZE * 2, STORE_CHUNK_ELEM * sizeof(DtypeForCpuOp));
    }
    shared_ptr<ChunkPool> chunk_pool;
};

template<typename T>
class ChunkGuard {
public:
    ChunkGuard<T>(shared_ptr<ChunkPool> chunk_pool_, T*& pointer) :
        chunk_pool(chunk_pool_)
    {
        id = chunk_pool->get_chunk_id();
        pointer = (T*) chunk_pool->chunks[id];
    }
    ~ChunkGuard<T>() {
        chunk_pool->return_chunk_id(id);
    }
private:
    int id;
    shared_ptr<ChunkPool> chunk_pool;
};


class TrustedChunkManager {
public:
    static TrustedChunkManager& getInstance() {
        static TrustedChunkManager instance;
        return instance;
    }
    TrustedChunkManager(TrustedChunkManager const&) = delete;
    void operator=(TrustedChunkManager const&) = delete;

    IdT GetNewId() {
        return id_counter++;
    }

    const int start_idx = 1000;

    void StoreChunk(IdT id, void* src_chunk, int num_byte) {
        int num_byte_enc_chunk = CalcEncDataSize(0, num_byte);
        #ifdef PRINT_CHUNK_INFO
            printf("num in byte %d, ", num_byte);
        #endif
        SgxEncT* enc_chunk = (SgxEncT*) get_untrusted_mem(id, num_byte_enc_chunk);
        DtypeForCpuOp* src_float = (DtypeForCpuOp*) src_chunk;
        encrypt((uint8_t *) src_chunk,
                num_byte,
                (uint8_t *) (&(enc_chunk->payload)),
                (sgx_aes_gcm_128bit_iv_t *)(&(enc_chunk->reserved)),
                (sgx_aes_gcm_128bit_tag_t *)(&(enc_chunk->payload_tag)));
        // DtypeForCpuOp* dst_chunk = (DtypeForCpuOp*)malloc(num_byte);
        // GetChunk(id, dst_chunk, num_byte);
        // uint8_t* blind_chunk;
        // ChunkGuard<uint8_t> guard(blind_chunks, blind_chunk);
        // decrypt((uint8_t *) (&(enc_chunk->payload)),
        //         num_byte,
        //         (uint8_t *) dst_chunk,
        //         (sgx_aes_gcm_128bit_iv_t  *)(&(enc_chunk->reserved)),
        //         (sgx_aes_gcm_128bit_tag_t *)(&(enc_chunk->payload_tag)),
        //         (uint8_t *) blind_chunk);
        // src_float = (DtypeForCpuOp*) dst_chunk;
        // free(dst_chunk);
    }

    void GetChunk(IdT id, void* dst_chunk, int num_byte) {
        #ifdef PRINT_CHUNK_INFO
            printf("GetChunk, id %ld, num byte %d\n", id, num_byte);
        #endif
        int num_byte_enc_chunk = CalcEncDataSize(0, num_byte);
        uint8_t* blind_chunk;
        ChunkGuard<uint8_t> guard(blind_chunks, blind_chunk);
        SgxEncT* enc_chunk = (SgxEncT*) get_untrusted_mem(id, num_byte_enc_chunk);
        decrypt((uint8_t *) (&(enc_chunk->payload)),
                num_byte,
                (uint8_t *) dst_chunk,
                (sgx_aes_gcm_128bit_iv_t  *)(&(enc_chunk->reserved)),
                (sgx_aes_gcm_128bit_tag_t *)(&(enc_chunk->payload_tag)),
                (uint8_t *) blind_chunk);
        DtypeForCpuOp* src_float = (DtypeForCpuOp*) dst_chunk;
    }

protected:
    TrustedChunkManager() {
        max_num_byte_plain_chunk = STORE_CHUNK_ELEM * sizeof(DtypeForCpuOp);
        max_num_byte_enc_chunk = CalcEncDataSize(0, max_num_byte_plain_chunk);

        blind_chunks = make_shared<ChunkPool>(THREAD_POOL_SIZE, max_num_byte_plain_chunk);
    }

    void* get_untrusted_mem(IdT id, int num_byte) {
        void* dst_buf;
        bool is_diff_size = false;
        auto it = untrusted_mem_holder.begin();
        auto end = untrusted_mem_holder.end();
        int prev_num_byte;
        {
            std::unique_lock <std::mutex> lock(address_mutex);
            it = untrusted_mem_holder.find(id);
            end = untrusted_mem_holder.end();
        }
        if (it == end) {
            #ifdef PRINT_CHUNK_INFO
                printf("alloc new mem id %u byte %d\n", id, num_byte);
            #endif
            allocate_in_untrusted(&dst_buf, num_byte);
            {
                std::unique_lock<std::mutex> lock(address_mutex);
                untrusted_mem_holder[id] = std::make_pair(dst_buf, num_byte);
            }
        } else {
            std::unique_lock<std::mutex> lock(address_mutex);
            std::tie(dst_buf, prev_num_byte) = untrusted_mem_holder[id];
            if (prev_num_byte != num_byte) {
                is_diff_size = true;
            }
        }
        if (is_diff_size) {
            // Usually cause by passing length instead of num_byte, * sizeof(DtypeForCpuOp)
			printf("id=%u\n",id);
            printf("A id has assigned with multiple size: original: %d, now: %d\n", prev_num_byte, num_byte);
            throw std::invalid_argument("A id has assigned with multiple size.");
        }
        return dst_buf;
    }

    const int size_chunk_pool = THREAD_POOL_SIZE;
    int max_num_byte_plain_chunk;
    int max_num_byte_enc_chunk;
    std::atomic<int> id_counter;
    std::mutex address_mutex;
    std::shared_ptr<ChunkPool> blind_chunks;
    std::unordered_map<int, std::pair<void*, int>> untrusted_mem_holder;
};

class TrustedChunkManagerUint8:TrustedChunkManager {
public:
    static TrustedChunkManagerUint8& getInstance() {
        static TrustedChunkManagerUint8 instance;
        return instance;
    }
    TrustedChunkManagerUint8(TrustedChunkManagerUint8 const&) = delete;
    void operator=(TrustedChunkManagerUint8 const&) = delete;

    IdT GetNewId() {
        return id_counter++;
    }

    const int start_idx = 1000;

    void StoreChunk(IdT id, void* src_chunk, int num_byte) {
        int num_byte_enc_chunk = CalcEncDataSize(0, num_byte);
        // #ifdef PRINT_CHUNK_INFO
        //     printf("num in byte %d, ", num_byte);
        // #endif
        // printf("TrustedChunkManagerUint8 StoreChunk id %llu, num_byte %d\n", id, num_byte);
        SgxEncT* enc_chunk = (SgxEncT*) get_untrusted_mem(id, num_byte_enc_chunk);
        DtypeForQuant* src_float = (DtypeForQuant*) src_chunk;
        encrypt((uint8_t *) src_chunk,
                num_byte,
                (uint8_t *) (&(enc_chunk->payload)),
                (sgx_aes_gcm_128bit_iv_t *)(&(enc_chunk->reserved)),
                (sgx_aes_gcm_128bit_tag_t *)(&(enc_chunk->payload_tag)));
    }

    void GetChunk(IdT id, void* dst_chunk, int num_byte) {
        #ifdef PRINT_CHUNK_INFO
            printf("GetChunk, id %ld, num byte %d\n", id, num_byte);
        #endif
        // printf("TrustedChunkManagerUint8 GetChunk id %llu, num_byte %d\n", id, num_byte);
        int num_byte_enc_chunk = CalcEncDataSize(0, num_byte);
        uint8_t* blind_chunk;
        ChunkGuard<uint8_t> guard(blind_chunks, blind_chunk);
        SgxEncT* enc_chunk = (SgxEncT*) get_untrusted_mem(id, num_byte_enc_chunk);
        decrypt((uint8_t *) (&(enc_chunk->payload)),
                num_byte,
                (uint8_t *) dst_chunk,
                (sgx_aes_gcm_128bit_iv_t  *)(&(enc_chunk->reserved)),
                (sgx_aes_gcm_128bit_tag_t *)(&(enc_chunk->payload_tag)),
                (uint8_t *) blind_chunk);
        DtypeForQuant* src_float = (DtypeForQuant*) dst_chunk;
    }

protected:
    TrustedChunkManagerUint8() {
        max_num_byte_plain_chunk = STORE_CHUNK_ELEM * sizeof(DtypeForQuant);
        max_num_byte_enc_chunk = CalcEncDataSize(0, max_num_byte_plain_chunk);
        blind_chunks = make_shared<ChunkPool>(THREAD_POOL_SIZE, max_num_byte_plain_chunk);
    }

    const int size_chunk_pool = THREAD_POOL_SIZE;
    int max_num_byte_plain_chunk;
    int max_num_byte_enc_chunk;
    std::atomic<int> id_counter;
    std::mutex address_mutex;
    std::shared_ptr<ChunkPool> blind_chunks;
    std::unordered_map<int, std::pair<void*, int>> untrusted_mem_holder;
};


template <typename Func>
void run_all_chunks(Func chunk_op, int num_elem_in_chunk, int num_elem) {
    int start_chunk;
    for (start_chunk = 0; start_chunk + num_elem_in_chunk <= num_elem; start_chunk += num_elem_in_chunk) {
        chunk_op(start_chunk, num_elem_in_chunk);
    }
    if (start_chunk < num_elem) chunk_op(start_chunk, num_elem - start_chunk);
}

template <typename Func>
void run_all_chunks_for_maxpool(Func chunk_op, size_t num_elem_in_chunk, size_t num_elem, size_t num_elem_out, size_t inputhw, size_t outputhw) {
    size_t start_chunk;
    for (start_chunk = 0; start_chunk + num_elem_in_chunk <= num_elem; start_chunk += num_elem_in_chunk) {
        chunk_op(start_chunk, num_elem_in_chunk, num_elem_out);
    }
    
    size_t remain_size = num_elem - start_chunk;
    if (start_chunk < num_elem) chunk_op(start_chunk, remain_size, (remain_size/inputhw)*outputhw);
}

class SecretTen {
public:
    SecretTen() {}
    SecretTen(IdT TenId_, DimsT* Dims_) : TenId(TenId_), Dims(*Dims_) { Init(); }
    ~SecretTen() { 
        for (auto& it: PrgStateHolder) free(it.second);
    }

    int GetNumElem() { return Dims.dim0 * Dims.dim1 * Dims.dim2 * Dims.dim3; }
    int GetSizeInByte() { return GetNumElem() * sizeof(DtypeForCpuOp); }

    void Init() {
        DtypeForCpuOp* store_chunk;
        ChunkGuard<DtypeForCpuOp> guard(StoreChunkPool::GetChunkPool(), store_chunk);
        auto& chunk_manager = TrustedChunkManager::getInstance();

        auto chunk_op = [&](int start, int num_elem_in_op) {
            int chunk_id = chunk_manager.GetNewId();
            ChunkIds.push_back(chunk_id);
            // printf("num_elem_in_op %d, ", num_elem_in_op);
            chunk_manager.StoreChunk(chunk_id, store_chunk, num_elem_in_op * sizeof(DtypeForCpuOp));
        };
        run_all_chunks(chunk_op, STORE_CHUNK_ELEM, GetNumElem());
    }

    int GetChunkId(int start) {
        if (start >= GetNumElem()) {
            printf("The start exceed the size of the tensor.\n");
            throw std::invalid_argument("The start exceed the size of the tensor.");
        }
        // printf("SecretTen.GetChunkId ChunkIds (");
        // for (int i=0; i<ChunkIds.size(); i++){
        //     printf("%d, ", ChunkIds[i]);
        // }
        // printf(")\n");
        return ChunkIds[start / STORE_CHUNK_ELEM];
    }

    void SetTen(DtypeForCpuOp* Arr) {
        auto& chunk_manager = TrustedChunkManager::getInstance();
        auto chunk_op = [&](int start, int num_elem_in_op) {
            int chunk_id = GetChunkId(start);
            DtypeForCpuOp* src_arr = Arr + start;
            chunk_manager.StoreChunk(chunk_id, src_arr, num_elem_in_op * sizeof(DtypeForCpuOp));
        };
        run_all_chunks(chunk_op, STORE_CHUNK_ELEM, GetNumElem());
    }

    void GetTen(DtypeForCpuOp* Arr) {
        auto& chunk_manager = TrustedChunkManager::getInstance();
        auto chunk_op = [&](int start, int num_elem_in_op) {
            int chunk_id = GetChunkId(start);
            DtypeForCpuOp* dst_arr = Arr + start;
            chunk_manager.GetChunk(chunk_id, dst_arr, num_elem_in_op * sizeof(DtypeForCpuOp));
        };
        run_all_chunks(chunk_op, STORE_CHUNK_ELEM, GetNumElem());
    }

    void SetSeed(uint64_t RawSeed) {
        SeedT seed;
        memset(seed, 0, sizeof(SeedT));
        auto TmpRawSeed = RawSeed;
        for (int i = 0; TmpRawSeed > 0; i++) {
            seed[i] = (uint8_t) (TmpRawSeed & ((1 << 9) - 1));
            TmpRawSeed >>= 8;
        }
        PrgStateHolder[RawSeed] = (aes_stream_state*)memalign(16, sizeof(aes_stream_state));
        InitPrgWithSeed(PrgStateHolder[RawSeed], seed);
    }

    IdT TenId;
    DimsT Dims;
    vector<int> ChunkIds;
    unordered_map<uint64_t, aes_stream_state*> PrgStateHolder;
};

unordered_map<IdT, shared_ptr<SecretTen>> SecretTenHolder;
unordered_map<IdT, shared_ptr<EigenTensor>> TensorHolder;

shared_ptr<SecretTen> GetTenById(IdT TenId) {
    return SecretTenHolder[TenId];
}

class SecretQuantTen {
public:
    SecretQuantTen() {}
    SecretQuantTen(IdT TenId_, DimsT* Dims_) : TenId(TenId_), Dims(*Dims_) { Init(); }
    ~SecretQuantTen() { 
        for (auto& it: PrgStateHolder) free(it.second);
    }

    int GetNumElem() { return Dims.dim0 * Dims.dim1 * Dims.dim2 * Dims.dim3; }
    int GetSizeInByte() { return GetNumElem() * sizeof(DtypeForQuant); }

    void Init() {
        DtypeForQuant* store_chunk;
        ChunkGuard<DtypeForQuant> guard(StoreChunkPool::GetChunkPool(), store_chunk);
        auto& chunk_manager = TrustedChunkManagerUint8::getInstance();

        auto chunk_op = [&](int start, int num_elem_in_op) {
            int chunk_id = chunk_manager.GetNewId();
            ChunkIds.push_back(chunk_id);
            // printf("num_elem_in_op %d, ", num_elem_in_op);
            chunk_manager.StoreChunk(chunk_id, store_chunk, num_elem_in_op * sizeof(DtypeForQuant));
        };
        run_all_chunks(chunk_op, STORE_CHUNK_ELEM, GetNumElem());
    }

    int GetChunkId(int start) {
        if (start >= GetNumElem()) {
            printf("The start exceed the size of the tensor.\n");
            throw std::invalid_argument("The start exceed the size of the tensor.");
        }
        return ChunkIds[start / STORE_CHUNK_ELEM];
    }

    void SetTen(DtypeForQuant* Arr) {
        auto& chunk_manager = TrustedChunkManagerUint8::getInstance();
        auto chunk_op = [&](int start, int num_elem_in_op) {
            int chunk_id = GetChunkId(start);
            DtypeForQuant* src_arr = Arr + start;
            chunk_manager.StoreChunk(chunk_id, src_arr, num_elem_in_op * sizeof(DtypeForQuant));
        };
        run_all_chunks(chunk_op, STORE_CHUNK_ELEM, GetNumElem());
    }

    void GetTen(DtypeForQuant* Arr) {
        auto& chunk_manager = TrustedChunkManagerUint8::getInstance();
        auto chunk_op = [&](int start, int num_elem_in_op) {
            int chunk_id = GetChunkId(start);
            DtypeForQuant* dst_arr = Arr + start;
            chunk_manager.GetChunk(chunk_id, dst_arr, num_elem_in_op * sizeof(DtypeForQuant));
        };
        run_all_chunks(chunk_op, STORE_CHUNK_ELEM, GetNumElem());
    }

    IdT TenId;
    DimsT Dims;
    vector<int> ChunkIds;
    unordered_map<uint64_t, aes_stream_state*> PrgStateHolder;
};

unordered_map<IdT, shared_ptr<SecretQuantTen>> SecretQuantTenHolder;
shared_ptr<SecretQuantTen> GetQuantTenById(IdT TenId) {
    return SecretQuantTenHolder[TenId];
}

unordered_map<uint64_t, DtypeForCpuOp> quantize_exp;

static inline float uint32_to_float(uint32_t x) {
    const union { uint32_t i; float d;  } u = { .i = UINT32_C(0x7F) << 23 | x >> 9  };
    return u.d - 1.0f;
}

static inline float float_to_uniform(uint32_t x) {
    const union { uint32_t i; float d;  } u = { .i = (((UINT32_C(0x7F) << 23) | x) << 2) >> 2 };
    return u.d - 1.0f;
}

// http://prng.di.unimi.it/
class Xoshiro256 {
public:
    Xoshiro256() {}
    Xoshiro256(uint64_t raw_seed) {
        set_seed(raw_seed);
    }

    void set_seed(uint64_t raw_seed) {
        s[0] = raw_seed;
    }

    static inline uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

    uint64_t next(void) {
        const uint64_t result = rotl(s[0] + s[3], 23) + s[0];

        const uint64_t t = s[1] << 17;

        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];

        s[2] ^= t;

        s[3] = rotl(s[3], 45);

        return result;
    }

    void rand_like(float* arr, uint64_t n_elem) {
        if (n_elem % 2 != 0) {
            printf("n_elem has to be even.\n");
            throw string("n_elem has to be even.");
        }
        for (int i = 0; i < n_elem; i+=2) {
            const uint64_t rnd = next();
            const uint32_t b = rnd & ((((uint64_t) 1) << 32) - 1);
            const uint32_t a = rnd >> 32;
            arr[i]   = uint32_to_float(a);
            arr[i+1] = uint32_to_float(b);
        }
    }

    uint64_t s[4] = {};
};

class Xoshiro128 {
public:
    Xoshiro128() {}
    Xoshiro128(uint64_t raw_seed) {
        set_seed(raw_seed);
    }

    void set_seed(uint64_t raw_seed) {
        s[0] = raw_seed;
    }

    static inline uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

    uint64_t next(void) {
        const uint64_t s0 = s[0];
        uint64_t s1 = s[1];
        const uint64_t result = rotl(s0 + s1, 17) + s0;

        s1 ^= s0;
        s[0] = rotl(s0, 49) ^ s1 ^ (s1 << 21); // a, b
        s[1] = rotl(s1, 28); // c

        return result;
    }

    uint64_t s[2] = {};
};

unordered_map<uint64_t, shared_ptr<Xoshiro256>> fast_rngs;
//unordered_map<uint64_t, shared_ptr<Xoshiro128>> fast_rngs;

shared_ptr<Xoshiro256> get_fast_rng(uint64_t tag) {
    if (fast_rngs.find(tag) == fast_rngs.end()) {
        fast_rngs[tag] = make_shared<Xoshiro256>(tag);
    }
    return fast_rngs[tag];
}

void quantize_stochastic(shared_ptr<SecretTen> src_ten, shared_ptr<SecretTen> dst_ten, uint64_t quantize_tag) {
    const int bits = 8;
    const int ebit = 8;
    const DtypeForCpuOp lower_limit = -pow(2, (bits - 1));
    const DtypeForCpuOp upper_limit = pow(2, (bits - 1)) - 1;

    auto& chunk_manager = TrustedChunkManager::getInstance();
    DtypeForCpuOp *store_chunk, *dst_store_chunk;
    ChunkGuard<DtypeForCpuOp> guard(StoreChunkPool::GetChunkPool(), store_chunk);
    ChunkGuard<DtypeForCpuOp> dst_guard(StoreChunkPool::GetChunkPool(), dst_store_chunk);
    //DtypeForCpuOp max_entry = 0;
    
	const __m256 neg8f = _mm256_set1_ps(-0.0f);
    __m256 tmp8f = _mm256_set1_ps(0.0f);

    auto get_max_chunk_op = [&](int start_store_chunk, int num_elem_in_store_chunk) {
        int chunk_id = src_ten->GetChunkId(start_store_chunk);
        chunk_manager.GetChunk(chunk_id, store_chunk, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));
        for(uint64_t i=0;i<num_elem_in_store_chunk;i+=8){
            const __m256 inp8f = _mm256_load_ps(&store_chunk[i]);
            const __m256 abs8f = _mm256_andnot_ps(neg8f, inp8f);
            const __m256 if_eq = _mm256_cmp_ps(inp8f, tmp8f, 0x0e);
            tmp8f = _mm256_blendv_ps(tmp8f, inp8f, if_eq);
        }
        //MapEigenVector src_vecmap(store_chunk, num_elem_in_store_chunk);
        //max_entry = std::max(max_entry, src_vecmap.cwiseAbs().maxCoeff());
    };
    run_all_chunks(get_max_chunk_op, STORE_CHUNK_ELEM, src_ten->GetNumElem());
    _mm256_stream_ps(dst_store_chunk, tmp8f);
    for(int i=4;i>0;i=i>>1){
        copy(dst_store_chunk+i,dst_store_chunk+2*i,dst_store_chunk+8);
        const __m256 inp8f = _mm256_load_ps(dst_store_chunk);
        const __m256 inp8f2 = _mm256_load_ps(&dst_store_chunk[8]);
        const __m256 if_eq = _mm256_cmp_ps(inp8f, inp8f2, 0x0e);
        const __m256 res8f = _mm256_blendv_ps(inp8f2, inp8f, if_eq);
        _mm256_stream_ps(dst_store_chunk, res8f);
    }

    if(1){
        dst_store_chunk[0] = (dst_store_chunk[0] == 0) ? 0: floor(log2(dst_store_chunk[0]));
        const __m256 inp8f = _mm256_load_ps(dst_store_chunk);
        //tmp8f = _mm256_set1_ps(pow(-2, (ebit - 1)));
        //__m256 if_gt = _mm256_cmp_ps(inp8f, tmp8f, 0x0e);
        //__m256 res8f = _mm256_blendv_ps(tmp8f, inp8f, if_gt); 
        tmp8f = _mm256_set1_ps(pow(2, (ebit - 1)) - 1);
        __m256 if_gt = _mm256_cmp_ps(inp8f, tmp8f, 0x0e);
        tmp8f = _mm256_blendv_ps(inp8f, tmp8f, if_gt);
        _mm256_stream_ps(dst_store_chunk, tmp8f);
    }
    DtypeForCpuOp exp = dst_store_chunk[0];

  //  DtypeForCpuOp exp = (max_entry == 0) ? 0 : floor(log2(max_entry));
  //  exp = std::min(std::max(exp, (DtypeForCpuOp) pow(-2, (ebit - 1))), (DtypeForCpuOp) pow(2, (ebit - 1) - 1));
    quantize_exp[quantize_tag] = exp;
    DtypeForCpuOp enlarge_factor = pow(2, -exp + (bits - 2));

    auto& xor_rnd = *get_fast_rng(quantize_tag);

    auto store_chunk_op = [&](int start_store_chunk, int num_elem_in_store_chunk) {
        chunk_manager.GetChunk(src_ten->GetChunkId(start_store_chunk), store_chunk, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));
        chunk_manager.GetChunk(dst_ten->GetChunkId(start_store_chunk), dst_store_chunk, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));

        auto chunk_op = [&](int start, int num_elem_in_op) {
            float *input = store_chunk + start;
            float *output = dst_store_chunk + start;
            xor_rnd.rand_like(output, num_elem_in_op);
            for(uint64_t i=0;i<num_elem_in_op;i+=8){
				tmp8f = _mm256_set1_ps(enlarge_factor); 
                const __m256 inp8f = _mm256_load_ps(&input[i]);
                const __m256 out8f = _mm256_load_ps(&output[i]);
                const __m256 mul8f  = _mm256_mul_ps(inp8f, tmp8f);  
                const __m256 add8f = _mm256_add_ps(mul8f, out8f);  
                const __m256 flo8f = _mm256_floor_ps(add8f);
                tmp8f = _mm256_set1_ps(lower_limit);
                __m256 if_gt = _mm256_cmp_ps(flo8f, tmp8f, 0x0e);
                __m256 res8f = _mm256_blendv_ps(tmp8f, flo8f, if_gt);
                tmp8f = _mm256_set1_ps(upper_limit);
                if_gt = _mm256_cmp_ps(res8f, tmp8f, 0x0e);
                res8f = _mm256_blendv_ps(res8f, tmp8f, if_gt);
                _mm256_stream_ps(&output[i], res8f);
            }
            //MapEigenTensor in_map = MapEigenTensor(input, 1, 1, 1, num_elem_in_op);
            //MapEigenTensor out_map = MapEigenTensor(output, 1, 1, 1, num_elem_in_op);
            //out_map = (in_map * enlarge_factor + out_map).floor().cwiseMax(lower_limit).cwiseMin(upper_limit);
        };
        run_all_chunks(chunk_op, WORK_CHUNK_ELEM, num_elem_in_store_chunk);
		//add
		chunk_manager.StoreChunk(dst_ten->GetChunkId(start_store_chunk), dst_store_chunk, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));
		//add
    };
    run_all_chunks(store_chunk_op, STORE_CHUNK_ELEM, src_ten->GetNumElem());
}

void dequantize_stochastic(shared_ptr<SecretTen> src_ten, shared_ptr<SecretTen> dst_ten,
        uint64_t x_tag, uint64_t y_tag) {
    const int bits = 8;
    DtypeForCpuOp x_exp = quantize_exp[x_tag];
    DtypeForCpuOp y_exp = quantize_exp[y_tag];

    auto& chunk_manager = TrustedChunkManager::getInstance();
    DtypeForCpuOp *store_chunk, *dst_store_chunk;
    ChunkGuard<DtypeForCpuOp> guard(StoreChunkPool::GetChunkPool(), store_chunk);
    ChunkGuard<DtypeForCpuOp> dst_guard(StoreChunkPool::GetChunkPool(), dst_store_chunk);

    auto store_chunk_op = [&](int start_store_chunk, int num_elem_in_store_chunk) {
        chunk_manager.GetChunk(src_ten->GetChunkId(start_store_chunk), store_chunk, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));
        chunk_manager.GetChunk(dst_ten->GetChunkId(start_store_chunk), dst_store_chunk, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));
        MapEigenTensor src_map = MapEigenTensor(store_chunk, 1, 1, 1, num_elem_in_store_chunk);
        MapEigenTensor dst_map = MapEigenTensor(dst_store_chunk, 1, 1, 1, num_elem_in_store_chunk);
        DtypeForCpuOp shrink_factor = pow(2, x_exp - (bits - 2) + y_exp - (bits - 2));

        dst_map = src_map * shrink_factor;
    };
    run_all_chunks(store_chunk_op, STORE_CHUNK_ELEM, src_ten->GetNumElem());
}

DtypeForCpuOp* get_small_chunk(
        shared_ptr<SecretTen> tensor,
        vector<std::pair<shared_ptr<SecretTen>, DtypeForCpuOp*>>& small_chunks) {

    int size_in_byte = tensor->GetSizeInByte();
    DtypeForCpuOp* arr = (DtypeForCpuOp*) memalign(64, size_in_byte);
    auto& chunk_manager = TrustedChunkManager::getInstance();
    chunk_manager.GetChunk(tensor->GetChunkId(0), arr, size_in_byte);
    small_chunks.emplace_back(tensor, arr);
    return arr;
}

void store_small_chunks(vector<std::pair<shared_ptr<SecretTen>, DtypeForCpuOp*>>& small_chunks) {
    for (auto& x : small_chunks) {
        auto tensor = x.first;
        auto arr = x.second;
        auto& chunk_manager = TrustedChunkManager::getInstance();
        int size_in_byte = tensor->GetSizeInByte();
        chunk_manager.StoreChunk(tensor->GetChunkId(0), arr, size_in_byte);
        free(arr);
    }
}


class BatchnormBuffer {
public:
    BatchnormBuffer(){}
    BatchnormBuffer(IdT FunId_) : FunId(FunId_) {
        NumBatchesTrackedArr = 0;
        BackwardState = false;
    }

    ~BatchnormBuffer() = default;

    void init(
            IdT input, IdT output, IdT gamma, IdT beta,
            // IdT der_input, IdT der_output, IdT der_gamma, IdT der_beta,
            IdT run_mean, IdT run_var, IdT cur_mean, IdT cur_var,
            IdT mu,
            uint32_t batch_, uint32_t channel_, uint32_t height_, uint32_t width_,
            int affine_, int is_cumulative_, float momentum_, float epsilon_) {

        input_tensor = GetTenById(input);
        output_tensor = GetTenById(output);
        // der_input_tensor = GetTenById(der_input);
        // der_output_tensor = GetTenById(der_output);
        mu_tensor = GetTenById(mu);

        // size = num_channel * sizeof(byte)
        gamma_tensor = GetTenById(gamma);
        beta_tensor = GetTenById(beta);
        // der_gamma_tensor = GetTenById(der_gamma);
        // der_beta_tensor = GetTenById(der_beta);
        run_mean_tensor = GetTenById(run_mean);
        run_var_tensor = GetTenById(run_var);
        cur_mean_tensor = GetTenById(cur_mean);
        cur_var_tensor = GetTenById(cur_var);
        

        batch = batch_;
        channel = channel_;
        height = height_;
        width = width_;
        Affine = affine_;
        momentum = momentum_;
        epsilon = epsilon_;
        is_cumulative = is_cumulative_;

        num_elem_per_sample = channel * height * width;
        // BCHW
        num_elem_in_channel = height * width;
        total_n = height * width * batch;
        
        default_num_batches_per_chunk = std::min(STORE_CHUNK_ELEM, input_tensor->GetNumElem()) / num_elem_per_sample;
        default_num_rows_per_chunk = std::min(STORE_CHUNK_ELEM, input_tensor->GetNumElem()) / num_elem_in_channel;
        // printf("Default batches per chunk %d, rows per chunk %d\n", default_num_batches_per_chunk, default_num_rows_per_chunk);
        if (STORE_CHUNK_ELEM % num_elem_in_channel != 0)  {
            printf(
                "STORE_CHUNK_ELEM %% num_elem_in_channel != 0, STORE_CHUNK_ELEM %d, num_elem_in_channel %d, left %d\n", 
                STORE_CHUNK_ELEM, num_elem_in_channel, STORE_CHUNK_ELEM % num_elem_in_channel
            );
            return;
        }
    }

    DtypeForCpuOp get_fraction_bag(int num_elem_in_chunk) {
        int batch_in_chunk = num_elem_in_chunk / num_rows;
        return ((DtypeForCpuOp) batch_in_chunk / batch);
    }

    int get_num_batches_per_chunk(int num_elem_in_chunk) {
        return num_elem_in_chunk / num_rows;
    }

    void forward(int training) {
        Training = training;

        vector<std::pair<shared_ptr<SecretTen>, DtypeForCpuOp*>> small_chunks;

        auto& chunk_manager = TrustedChunkManager::getInstance();
        DtypeForCpuOp *data_chunk, *mu_chunk;
        ChunkGuard<DtypeForCpuOp> data_guard(StoreChunkPool::GetChunkPool(), data_chunk);
        ChunkGuard<DtypeForCpuOp> mu_guard(StoreChunkPool::GetChunkPool(), mu_chunk);

        // EigenMatrixMap data_mat(data_chunk, num_rows, default_num_batches_per_chunk);
        // EigenMatrixMap mu_mat(mu_chunk, num_rows, default_num_batches_per_chunk);

        DtypeForCpuOp *gamma_chunk = get_small_chunk(gamma_tensor, small_chunks);
        DtypeForCpuOp *beta_chunk = get_small_chunk(beta_tensor, small_chunks);
        DtypeForCpuOp *run_mean_chunk = get_small_chunk(run_mean_tensor, small_chunks);
        DtypeForCpuOp *run_var_chunk = get_small_chunk(run_var_tensor, small_chunks);
        DtypeForCpuOp *cur_mean_chunk = get_small_chunk(cur_mean_tensor, small_chunks);
        DtypeForCpuOp *cur_var_chunk = get_small_chunk(cur_var_tensor, small_chunks);

        int total_input_row_idx = 0;

        if (training) {
            
        } else {
            run_all_chunks([&](int start_store_chunk, int num_elem_in_store_chunk) {
                int chunk_size_in_byte = num_elem_in_store_chunk * sizeof(DtypeForCpuOp);
                chunk_manager.GetChunk(input_tensor->GetChunkId(start_store_chunk), data_chunk, chunk_size_in_byte);
                int num_rows_in_chunk = num_elem_in_store_chunk / num_elem_in_channel;
                MapMatRowMajor data_mat(data_chunk, num_rows_in_chunk, num_elem_in_channel);

                // printf("data_chunk\n");
                // for (auto i=0; i<num_elem_in_store_chunk; i++){
                //     printf("%.2f, ", data_chunk[i]);
                // }
                // printf("\n");
                
                // printf("data_mat Mat:\n");
                // for (auto r=0; r<num_rows_in_chunk; r++){
                //     for (auto c=0; c<num_elem_in_channel; c++)
                //         printf("%.2f ", data_mat(r,c));
                //     printf("\n");
                // }

                for (auto row_idx_in_chunk=0; row_idx_in_chunk<num_rows_in_chunk; row_idx_in_chunk++){
                    auto channel_idx = total_input_row_idx % channel;
                    auto data_block = data_mat.block(row_idx_in_chunk, 0, 1, num_elem_in_channel);
                    data_block = data_block.array() - run_mean_chunk[channel_idx];
                    if (Affine) {
                        data_block = (data_block.array() / sqrt(run_var_chunk[channel_idx]+1e-5)) * gamma_chunk[channel_idx] + beta_chunk[channel_idx];
                        // running_var here is actually 1 / sqrt(running_var)
                        // printf("var %f\n", run_var_chunk[i]);
                        // data_block = (data_block.array() * run_var_chunk[i]) * gamma_chunk[i] + beta_chunk[i];
                    } else {
                        printf("Check var first!!!!!!==================\n");
                    }
                    total_input_row_idx++;
                }

                // printf("Bias: ");
                // for (auto i=0; i<channel; i++)
                //     printf("%f ", beta_chunk[i]);
                // printf("\n");
                // printf("RunVar: ");
                // for (auto i=0; i<channel; i++)
                //     printf("%f ", run_var_chunk[i]);
                // printf("\n");

                // for(uint32_t i = 0; i < channel; i++) {
                //     auto data_block = data_mat.block(i * num_rows_in_channel, 0, num_rows_in_channel, num_batches_per_chunk);
                //     data_block = data_block.array() - run_mean_chunk[i];
                //     if (Affine) {
                //         data_block = (data_block.array() / sqrt(run_var_chunk[i]+1e-5)) * gamma_chunk[i] + beta_chunk[i];
                //         // running_var here is actually 1 / sqrt(running_var)
                //         // printf("var %f\n", run_var_chunk[i]);
                //         // data_block = (data_block.array() * run_var_chunk[i]) * gamma_chunk[i] + beta_chunk[i];
                //     } else {
                //         printf("Check var first!!!!!!==================\n");
                //         data_block = data_block / sqrt(run_var_chunk[i]);
                //     }
                // }

                chunk_manager.StoreChunk(output_tensor->GetChunkId(start_store_chunk), data_chunk, chunk_size_in_byte);
            }, STORE_CHUNK_ELEM, input_tensor->GetNumElem());
        }

        store_small_chunks(small_chunks);

        BackwardState = true;
    }

    void backward() {}
    
    IdT FunId;
    int batch;
    int channel;
    int height;
    int width;
    DtypeForCpuOp momentum;
    DtypeForCpuOp epsilon;

    bool is_cumulative;
    bool BackwardState;
    bool Affine;
    bool Training;

    int num_rows;
    int num_elem_per_sample, num_elem_in_channel;
    int total_n;
    int default_num_batches_per_chunk, default_num_rows_per_chunk;

    int NumBatchesTrackedArr = 0;

    shared_ptr<SecretTen> input_tensor;
    shared_ptr<SecretTen> output_tensor;
    shared_ptr<SecretTen> der_input_tensor;
    shared_ptr<SecretTen> der_output_tensor;
    shared_ptr<SecretTen> mu_tensor;
    shared_ptr<SecretTen> gamma_tensor;
    shared_ptr<SecretTen> beta_tensor;
    shared_ptr<SecretTen> der_gamma_tensor;
    shared_ptr<SecretTen> der_beta_tensor;
    shared_ptr<SecretTen> run_mean_tensor;
    shared_ptr<SecretTen> run_var_tensor;
    shared_ptr<SecretTen> cur_mean_tensor;
    shared_ptr<SecretTen> cur_var_tensor;
};


class SGXLinearBuffer {
public:
    SGXLinearBuffer(){}
    SGXLinearBuffer(IdT FunId_) : FunId(FunId_) {
    }

    ~SGXLinearBuffer() = default;

    void init(
            IdT input, IdT output, IdT weight, IdT bias,
            // IdT der_input, IdT der_output, IdT der_weight, IdT der_bias,
            uint32_t batch_, uint32_t input_size_, uint32_t output_size_) {

        input_tensor = GetTenById(input);
        output_tensor = GetTenById(output);
        // der_input_tensor = GetTenById(der_input);
        // der_output_tensor = GetTenById(der_output);

        // size = num_channel * sizeof(byte)
        weight_tensor = GetTenById(weight);
        bias_tensor = GetTenById(bias);
        // der_weight_tensor = GetTenById(der_weight);
        // der_bias_tensor = GetTenById(der_bias);

        batch = batch_;
        input_size = input_size_;
        output_size = output_size_;

        printf("SGXLinearBuffer initialized: Batch %d, input %d, output %d\n", batch, input_size, output_size);

        printf(
            "SGXLinearBuffer chunk_size %d, total_input %d, output_size %d, weight_size %d\n", 
            STORE_CHUNK_ELEM, input_size * batch, batch*output_size, input_size*output_size);
        printf("features per chunk %d\n", STORE_CHUNK_ELEM/input_size);

        default_num_batches_per_chunk = std::min(STORE_CHUNK_ELEM, input_tensor->GetNumElem()) / input_size;
        default_num_col_per_chunk = std::min(STORE_CHUNK_ELEM, weight_tensor->GetNumElem()) / input_size;
        
        if (STORE_CHUNK_ELEM % input_size != 0)  {
            float ratio = STORE_CHUNK_ELEM / input_size;
            printf("SGXLinearBuffer STORE_CHUNK_ELEM num_rows != 0\n");
            printf("Chunk_size %d / inpu_size %d = %f \n", STORE_CHUNK_ELEM, input_size, ratio);
            return;
        }
    }

    int get_num_batches_per_chunk(int num_elem_in_chunk) {
        return num_elem_in_chunk / input_size;
    }

    void forward() {
        printf("SGXDNN_Main SGXLinearBuffer forward\n");
        
        auto& chunk_manager = TrustedChunkManager::getInstance();
        DtypeForCpuOp *data_chunk, *weight_chunk, *output_chunk, *bias_chunk;
        ChunkGuard<DtypeForCpuOp> data_guard(StoreChunkPool::GetChunkPool(), data_chunk);
        ChunkGuard<DtypeForCpuOp> weight_guard(StoreChunkPool::GetChunkPool(), weight_chunk);
        ChunkGuard<DtypeForCpuOp> output_guard(StoreChunkPool::GetChunkPool(), output_chunk);
        ChunkGuard<DtypeForCpuOp> bias_guard(StoreChunkPool::GetChunkPool(), bias_chunk);

        // Default eigen matrix is ColMajor
        
        
        MapMatRowMajor bias_mat(bias_chunk, 1, output_size);
        MapMatRowMajor output_mat(output_chunk, batch, output_size);

        chunk_manager.GetChunk(output_tensor->GetChunkId(0), output_chunk, batch*output_size*sizeof(DtypeForCpuOp));
        chunk_manager.GetChunk(bias_tensor->GetChunkId(0), bias_chunk, output_size*sizeof(DtypeForCpuOp));
        
        run_all_chunks([&](int data_start_store_chunk, int data_num_elem_in_store_chunk) {
            int data_chunk_size_in_byte = data_num_elem_in_store_chunk * sizeof(DtypeForCpuOp);
            int num_features_in_chunk = data_num_elem_in_store_chunk / input_size;
            MapMatRowMajor data_mat(data_chunk, num_features_in_chunk, input_size);
            int data_start_idx = data_start_store_chunk / input_size;
            chunk_manager.GetChunk(input_tensor->GetChunkId(data_start_store_chunk), data_chunk, data_chunk_size_in_byte);
            // printf("SGXDNN_Main Input array: ");
            // for (auto i=0; i<10; i++){
            //     printf("%f ", data_chunk[i]);
            // }
            // printf("\n");
            // printf("SGXDNN_Main Input mat: ");
            // for (auto i=0; i<10; i++){
            //     printf("%f ", data_mat(0,i));
            // }
            // printf("\n");
            run_all_chunks([&](int weight_start_store_chunk, int weight_num_elem_in_store_chunk) {
                
                int weight_chunk_size_in_byte = weight_num_elem_in_store_chunk * sizeof(DtypeForCpuOp);
                int num_classes_in_chunk = weight_num_elem_in_store_chunk / input_size;
                int class_start_idx = weight_start_store_chunk / input_size;
                // printf("data start %d, end %d. weight start %d, end %d\n",
                //     data_start_idx, data_start_idx+num_features_in_chunk,
                //     class_start_idx, class_start_idx+num_classes_in_chunk);
                MapMatRowMajor weight_mat(weight_chunk, num_classes_in_chunk, input_size);
                // printf("Rows %d, cols %d\n", num_features_in_chunk, num_classes_in_chunk);
                chunk_manager.GetChunk(weight_tensor->GetChunkId(weight_start_store_chunk), weight_chunk, weight_chunk_size_in_byte);

                // printf("SGXDNN_Main Weight array: ");
                // for (auto i=0; i<10; i++){
                //     printf("%f ", weight_chunk[i]);
                // }
                // printf("\n");
                // printf("SGXDNN_Main Weight mat: ");
                // for (auto i=0; i<10; i++){
                //     printf("%f ", weight_mat(0,i));
                // }
                // printf("\n");

                // printf("new rows %d, cols %d, size %d\n", output_mat.rows(), output_mat.cols(), output_mat.size());
                // printf("%d %d, %d %d\n", batch, output_size, num_features_in_chunk, num_classes_in_chunk);
                auto output_block = output_mat.block(data_start_idx, class_start_idx, num_features_in_chunk, num_classes_in_chunk);
                // printf("data_mat: rows %d, cols %d, size %d\n", data_mat.rows(), data_mat.cols(), data_mat.size());
                // printf("weight_mat: rows %d, cols %d, size %d\n", weight_mat.rows(), weight_mat.cols(), weight_mat.size());
                // printf("weight_mat.transpose(): rows %d, cols %d, size %d\n", weight_mat.transpose().rows(), weight_mat.transpose().cols(), weight_mat.transpose().size());
                // printf("output_mat: rows %d, cols %d, size %d\n", output_block.rows(), output_block.cols(), output_block.size());
                output_block.array() = data_mat * weight_mat.transpose();
                
            }, STORE_CHUNK_ELEM, weight_tensor->GetNumElem());
        }, STORE_CHUNK_ELEM, input_tensor->GetNumElem());

        // printf("SGXDNN_Main Nobias Output mat: ");
        // for (auto i=0; i<10; i++){
        //     printf("%f ", output_mat(0,i));
        // }
        // printf("\n");
        // printf("SGXDNN_Main Bias mat: ");
        // for (auto i=0; i<10; i++){
        //     printf("%f ", bias_mat(0,i));
        // }
        // printf("\n");

        // auto output_block = output_mat.block(0, 0, 1, output_size);
        // cannot declare output_block outside the loop
        for (auto i=0; i<batch; i++){
            auto output_block = output_mat.block(i, 0, 1, output_size);
            // if ( i==0 ){
            //     printf("Debug bias: ");
            //     printf("output %f + bias %f", output_block(0,0), bias_mat(0,0));
            // }
            output_block.array() = output_block + bias_mat;
            // if (i==0){
            //     printf(" = %f, raw mat %f", output_block(0,0), output_mat(0,0));
            //     printf("\n");
            // }
            // printf("Debug raw mat %f\n", output_mat(0,0));
        }
        
        // printf("SGXDNN_Main Output mat: ");
        // for (auto i=0; i<10; i++){
        //     printf("%f ", output_mat(1,i));
        // }
        // printf("\n");
        chunk_manager.StoreChunk(output_tensor->GetChunkId(0), output_chunk, batch*output_size*sizeof(DtypeForCpuOp));

    }

    void backward() {}
    
    IdT FunId;
    int batch;
    int input_size;
    int output_size;

    int num_rows;
    int num_rows_in_channel;
    int total_n;
    int default_num_batches_per_chunk, default_num_col_per_chunk;
    int features_per_chunk, classes_per_chunk;

    int NumBatchesTrackedArr = 0;

    shared_ptr<SecretTen> input_tensor;
    shared_ptr<SecretTen> output_tensor;
    shared_ptr<SecretTen> der_input_tensor;
    shared_ptr<SecretTen> der_output_tensor;
    shared_ptr<SecretTen> weight_tensor;
    shared_ptr<SecretTen> bias_tensor;
    shared_ptr<SecretTen> der_weight_tensor;
    shared_ptr<SecretTen> der_bias_tensor;
};


template <typename Func>
void run_all_chunks_conv(Func chunk_op, int num_elem_in_chunk, int num_elem) {
    int start_chunk;
    for (start_chunk = 0; start_chunk + num_elem_in_chunk <= num_elem; start_chunk += num_elem_in_chunk ) {
        chunk_op(start_chunk, num_elem_in_chunk, start_chunk + num_elem_in_chunk == num_elem);
    }
    if (start_chunk < num_elem) chunk_op(start_chunk, num_elem - start_chunk, true);
}

class SGXConvBuffer {
public:
    SGXConvBuffer(){}
    SGXConvBuffer(IdT FunId_) : FunId(FunId_) {
    }

    ~SGXConvBuffer() = default;

    void init(
            IdT input, IdT output, IdT weight, IdT bias,
            // IdT der_input, IdT der_output, IdT der_weight, IdT der_bias,
            uint32_t batch_, uint32_t input_h_, uint32_t input_w_, uint32_t input_c_,
            uint32_t output_h_, uint32_t output_w_, uint32_t output_c_,
            uint32_t kernel_, uint32_t padding_, uint32_t stride_) {
        #ifdef PRINT_CONV_INIT_INFO
            printf("SGX Conv Buffer init\n", input);
        #endif

        input_tensor = GetTenById(input);
        output_tensor = GetTenById(output);
        // der_input_tensor = GetTenById(der_input);
        // der_output_tensor = GetTenById(der_output);

        // size = num_channel * sizeof(byte)
        weight_tensor = GetTenById(weight);
        // der_weight_tensor = GetTenById(der_weight);
        bias_tensor = GetTenById(bias);
        // der_bias_tensor = GetTenById(der_bias);

        batch = batch_;
        input_h = input_h_; input_w = input_w_; input_c = input_c_;
        output_h = output_h_; output_w = output_w_; output_c = output_c_;
        kernel = kernel_; padding = padding_; stride = stride_;

        input_row_size = input_w * input_c; one_batch_input_size = input_h * input_row_size;
        if (STORE_CHUNK_ELEM % (input_row_size*stride) != 0){
            printf("STORE_CHUNK_ELEM %d cannot divide input_row_size*stride %d*%d, STORE_CHUNK_ELEM %% input_row_size=%d\n", 
            STORE_CHUNK_ELEM, input_row_size, stride, STORE_CHUNK_ELEM % (input_row_size*stride));
        }
        assert (STORE_CHUNK_ELEM % input_row_size == 0);
        int input_rows_per_chunk = STORE_CHUNK_ELEM / input_row_size;
        input_elem_fetch_per_chunk = input_rows_per_chunk * input_row_size;
        int num_input_per_chunk = input_rows_per_chunk / input_h;
        int ramain_rows_per_chunk = input_rows_per_chunk % input_h;
        #ifdef PRINT_CONV_INIT_INFO
            printf(
                "ChunkElem %d, row_size %d, rows %d, remain %d, fetch_elem %d\n", 
                STORE_CHUNK_ELEM, input_row_size, input_rows_per_chunk, STORE_CHUNK_ELEM % input_row_size, input_elem_fetch_per_chunk);
        #endif

        // pytorch weight shape is [output_c, input_c, kernel, kernel]
        #ifdef PRINT_CONV_INIT_INFO
            printf(
                "SGXConvBuffer initialized: Batch %d, input [%d,%d,%d], output [%d,%d,%d], weight [%d,%d,%d,%d], ", 
                batch, input_c, input_h, input_w, output_c, output_h, output_w,
                output_c, input_c, kernel, kernel);
            printf("kernel %d, padding %d, stride %d\n", kernel, padding, stride);
            printf("chunk_size %d, single batch size %d\n", STORE_CHUNK_ELEM, input_c*input_h*input_w);
        #endif

        patch_size = kernel * kernel * input_c;
        // Compute max im2col patches
        assert (STORE_CHUNK_ELEM > patch_size);
        max_im2col_patches_per_chunk = STORE_CHUNK_ELEM / patch_size;
        int total_im2col_patches = batch * output_h * output_w;
        max_im2col_patches_per_chunk = std::min(max_im2col_patches_per_chunk, total_im2col_patches);
        // Compute max output patches
        if (STORE_CHUNK_ELEM < output_c)
            printf("output channel (%d) is larger than STORE_CHUNK_ELEM\n", output_c);
        assert (STORE_CHUNK_ELEM >= output_c);
        if (STORE_CHUNK_ELEM % output_c != 0){
            printf("STORE_CHUNK_ELEM %d cannot divide output_channel %d, STORE_CHUNK_ELEM %% output_c=%d\n", 
                STORE_CHUNK_ELEM, output_c, STORE_CHUNK_ELEM % output_c);
        }
        assert (STORE_CHUNK_ELEM % output_c == 0);
        max_output_patches_per_chunk = STORE_CHUNK_ELEM / output_c;
        int total_output_patches = batch * output_h * output_w;
        max_output_patches_per_chunk = std::min(max_output_patches_per_chunk, total_output_patches);
        // Compute matrix mul rows
        max_matrix_mul_rows = std::min(max_im2col_patches_per_chunk, max_output_patches_per_chunk);
        // max_matrix_mul_rows = max_im2col_patches_per_chunk;
        // max_matrix_mul_rows = max_output_patches_per_chunk;
        
        im2col_patches_per_chunk = max_matrix_mul_rows * (max_im2col_patches_per_chunk / max_matrix_mul_rows);
        // output_patches_per_chunk = max_matrix_mul_rows * (max_output_patches_per_chunk / max_matrix_mul_rows);
        output_patches_per_chunk = max_output_patches_per_chunk;
        im2col_num_elem_in_chunk = im2col_patches_per_chunk * patch_size;
        output_num_elem_in_chunk = output_patches_per_chunk * output_c;

        max_weight_rows_per_chunk = max_im2col_patches_per_chunk;
        max_weight_elem_per_chunk = max_weight_rows_per_chunk * patch_size;

        #ifdef PRINT_CONV_INIT_INFO
            printf(
                "input  row_size %d, input_rows_per_chunk %d, inpu_chunk_size %d\n", 
                input_row_size, input_rows_per_chunk, input_row_size*input_rows_per_chunk);
            printf(
                "im2col row size %d, max_im2col_patches_per_chunk %d, max_matrix_mul_rows %d, im2col_patches_per_chunk %d, im2col_chunk_size %d\n", 
                patch_size, max_im2col_patches_per_chunk, max_matrix_mul_rows, im2col_patches_per_chunk, im2col_num_elem_in_chunk);
            printf(
                "output row size %d, max_output_patches_per_chunk %d, max_matrix_mul_rows %d, output_patches_per_chunk %d, output_chunk_size %d\n", 
                output_c, max_output_patches_per_chunk, max_matrix_mul_rows, output_patches_per_chunk, output_num_elem_in_chunk);
            printf(
                "weight row size %d, max_weight_rows_per_chunk %d, weight_chunk_size %d\n", 
                patch_size, max_weight_rows_per_chunk, max_weight_elem_per_chunk);
            printf("SGX Conv Buffer finish init\n");
        #endif

    }

    // int get_num_batches_per_chunk(int num_elem_in_chunk) {
    //     return num_elem_in_chunk / input_size;
    // }

    void forward() {
        // printf(
        //     "Secret Conv Forward, input (%d,%d,%d), output (%d,%d,%d), kernel %d, stride %d, padding %d\n",
        //     input_h, input_w, input_c,  output_h, output_w, output_c, kernel, stride,  padding
        // );
        #ifdef PRINT_RUN_TIME_INFO
            sgx_time_t total_start = get_time();
        #endif
        int output_width, output_height, stride_cols, stride_rows, filter_height, filter_width;
        int input_batches, input_height, input_width, input_depth;
        output_width = output_w; output_height = output_h;
        stride_cols = stride; stride_rows = stride;
        filter_height = kernel; filter_width = kernel;
        input_batches = batch; input_height = input_h; input_width = input_w; input_depth = input_c;

        sgx_time_t load_input_start_time, load_weight_start_time, save_output_start_time, im2col_construction_start_time, matrix_mul_start_time, forward_prepare_start_time;
        #ifdef PRINT_RUN_TIME_INFO
            forward_prepare_start_time = get_time();
        #endif
        const int filter_value_count = filter_width * filter_height * input_depth;
        if ((filter_value_count * sizeof(DtypeForCpuOp)) > STORE_CHUNK_ELEM){
            printf(
                "filter_value_count size is larger than STORE_CHUNK_ELEM, filter_value_count(%d) * DtypeForCpuOp(4) = %d > STORE_CHUNK_ELEM(%d)\n",
                filter_value_count, filter_value_count * sizeof(DtypeForCpuOp), STORE_CHUNK_ELEM
            );
        }
        assert((filter_value_count * sizeof(DtypeForCpuOp)) <= STORE_CHUNK_ELEM);
        auto& chunk_manager = TrustedChunkManager::getInstance();
        DtypeForCpuOp *data_chunk, *weight_chunk, *output_chunk, *im2col_chunk, *bias_chunk;
        ChunkGuard<DtypeForCpuOp> data_guard(StoreChunkPool::GetChunkPool(), data_chunk);
        ChunkGuard<DtypeForCpuOp> weight_guard(StoreChunkPool::GetChunkPool(), weight_chunk);
        ChunkGuard<DtypeForCpuOp> output_guard(StoreChunkPool::GetChunkPool(), output_chunk);
        ChunkGuard<DtypeForCpuOp> im2col_guard(StoreChunkPool::GetChunkPool(), im2col_chunk);
        ChunkGuard<DtypeForCpuOp> bias_guard(StoreChunkPool::GetChunkPool(), bias_chunk);
        // DtypeForCpuOp tmp_output_chunk[max_matrix_mul_rows * output_c];
        // DtypeForCpuOp* operate_weight_chunk = (DtypeForCpuOp*)malloc((max_weight_rows_per_chunk+3) * output_c * sizeof(DtypeForCpuOp));
        // DtypeForCpuOp operate_weight_chunk[(max_weight_rows_per_chunk+3) * output_c];
        // printf("Begin operate weight %p\n", operate_weight_chunk);
        // DtypeForCpuOp* redundent_weight_chunk = (DtypeForCpuOp*)malloc(3 * output_c * sizeof(DtypeForCpuOp));
        // DtypeForCpuOp redundent_weight_chunk[3 * output_c];
        // printf("Begin redundent weight %p\n", redundent_weight_chunk);
        DtypeForCpuOp* operate_weight_chunk, *redundent_weight_chunk;
        int redundant_weight_cnt = 0;
        // DtypeForCpuOp* tmp_output_chunk_end = tmp_output_chunk + max_matrix_mul_rows * output_c;
        // MapMatRowMajor tmp_output_mat(tmp_output_chunk, max_matrix_mul_rows, output_c);
        MapMatRowMajor bias_mat(bias_chunk, 1, output_c);
        chunk_manager.GetChunk(bias_tensor->GetChunkId(0), bias_chunk, output_c*sizeof(DtypeForCpuOp));
        chunk_manager.GetChunk(output_tensor->GetChunkId(0), output_chunk, output_num_elem_in_chunk*sizeof(DtypeForCpuOp));
        #ifdef PRINT_RUN_TIME_INFO
            sgx_time_t end = get_time();
            forward_prepare_time = get_elapsed_time(forward_prepare_start_time, end);
        #endif

        // Operate input chunks
        DtypeForCpuOp *operate_input_chunk, *redundant_input_chunk;
        int redundant_input_rows = 0, input_row_idx_total = 0;
        int filter_radius_height = filter_height / 2, filter_radius_width = filter_width / 2;
        // int reuse_num_elem = filter_radius_height * input_width * input_depth;
        int im2col_row_idx_in_chunk = 0, output_row_idx_in_chunk = 0, im2col_row_idx_total = 0, output_row_idx_total = 0;
        run_all_chunks_conv([&](int data_start_store_chunk, int data_num_elem_in_store_chunk, bool last_chunk) {
            
            int data_chunk_size_in_byte = data_num_elem_in_store_chunk * sizeof(DtypeForCpuOp);
            #ifdef PRINT_RUN_TIME_INFO
                load_input_start_time = get_time();
            #endif
            chunk_manager.GetChunk(input_tensor->GetChunkId(data_start_store_chunk), data_chunk, data_chunk_size_in_byte);
            
            
            
            #ifdef PRINT_CONV_INPUT_LOAD_CHUNK_INFO
                int data_chunk_start_row = data_start_store_chunk / input_row_size;
                int data_chunk_start_batch = data_chunk_start_row / input_height;
                int data_chunk_start_y_in_batch = data_chunk_start_row % input_height;
                int data_chunk_end_row = (data_start_store_chunk + data_num_elem_in_store_chunk) / input_row_size;
                int data_chunk_end_batch = data_chunk_end_row / input_height;
                int data_chunk_end_y_in_batch = data_chunk_end_row % input_height;
                printf(
                    "Load elem %d-%d, batch %d row %d to batch %d row %d\n",
                    data_start_store_chunk, data_start_store_chunk + data_num_elem_in_store_chunk,
                    data_chunk_start_batch, data_chunk_start_y_in_batch,
                    data_chunk_end_batch, data_chunk_end_y_in_batch
                );
            #endif

            int operate_input_rows = data_num_elem_in_store_chunk / input_row_size + redundant_input_rows;
            // printf("Before malloc operate_input_chunk\n");
            operate_input_chunk = (DtypeForCpuOp*)malloc(operate_input_rows*input_row_size*sizeof(DtypeForCpuOp));
            // printf("After malloc operate_input_chunk\n");
            if (redundant_input_rows > 0){
                // printf("Pre-saved redundant rows %d\n", redundant_input_rows);
                memcpy(operate_input_chunk, redundant_input_chunk, redundant_input_rows*input_row_size*sizeof(DtypeForCpuOp));
                free(redundant_input_chunk);
            }
            DtypeForCpuOp* operate_input_chunk_copy_start = operate_input_chunk + redundant_input_rows*input_row_size;
            memcpy(operate_input_chunk_copy_start, data_chunk, data_num_elem_in_store_chunk*sizeof(DtypeForCpuOp));
            MapMatRowMajor operate_data_mat(operate_input_chunk, operate_input_rows, input_row_size);
            // printf("Operate input chunk shape (%d, %d)\n", operate_input_rows, input_row_size);
            
            if (!last_chunk){
                int new_redundant_input_rows = std::max(0, filter_height - stride);
                // printf("new_redundant_input_rows %d\n", new_redundant_input_rows);
                int new_redundant_input_elem_num = new_redundant_input_rows*input_row_size;
                // printf("Before malloc redundant_input_chunk, size %d\n", new_redundant_input_elem_num);
                redundant_input_chunk = (DtypeForCpuOp*)malloc(new_redundant_input_elem_num*sizeof(DtypeForCpuOp));
                // printf("After malloc redundant_input_chunk\n");
                DtypeForCpuOp* new_redundant_input_start = data_chunk + data_num_elem_in_store_chunk - new_redundant_input_elem_num;
                // printf("Before copy, size %d\n", new_redundant_input_elem_num);
                memcpy(redundant_input_chunk, new_redundant_input_start, new_redundant_input_elem_num*sizeof(DtypeForCpuOp));
                // printf("After copy\n");
                redundant_input_rows = new_redundant_input_rows;
            }
            // load_input_time += clock() - load_input_start_time;

            // print input
            // printf("Input: \n");
            // printf("%f %f %f\n", data_chunk[0], data_chunk[1], data_chunk[2]);
            // for (int r=0; r<input_height; r++){
            //     for (int c=0; c<input_width; c++){
            //         printf("[");
            //         for (int d=0; d<input_depth; d++){
            //             int bias = r * input_width * input_depth +
            //                 c * input_depth + d;
            //             DtypeForCpuOp* p = data_chunk + bias;
            //             printf("%f  ", *p);
            //         }
            //         printf("],  ");
            //     }
            //     printf("\n");
            // }
            #ifdef PRINT_RUN_TIME_INFO
                load_input_time += get_elapsed_time(load_input_start_time, get_time());
            #endif

            // row_idx_in_chunk, col_idx is the centor pixel in the input feature
            // printf("Im2col: \n");
            int end_row_idx_in_operate_chunk, start_row_idx_in_operate_chunk;
            if (!last_chunk)
                end_row_idx_in_operate_chunk = operate_input_rows-filter_radius_height;
            else
                end_row_idx_in_operate_chunk = operate_input_rows;
            if (data_start_store_chunk == 0)
                start_row_idx_in_operate_chunk = 0;
            else
                start_row_idx_in_operate_chunk = filter_radius_height;
            // for (int relevant_up_row_idx_in_chunk=0; relevant_up_row_idx_in_chunk < end_row_idx_in_chunk; relevant_up_row_idx_in_chunk+=stride_rows){
            // for (int row_idx_in_chunk=start_row_idx_in_operate_chunk; row_idx_in_chunk < end_row_idx_in_operate_chunk; row_idx_in_chunk+=stride_rows){
            int row_idx_in_chunk = start_row_idx_in_operate_chunk;
            #ifdef PRINT_CONV_IM2COL_CONSTRUCT_INFO
                printf("New input chunk, start row %d, end row %d\n", start_row_idx_in_operate_chunk, end_row_idx_in_operate_chunk);
            #endif
            // printf("Tag2\n");
            while (row_idx_in_chunk < end_row_idx_in_operate_chunk){
                int batch_idx = input_row_idx_total / input_height;
                int row_idx_in_batch = input_row_idx_total % input_height;
                #ifdef PRINT_CONV_IM2COL_CONSTRUCT_INFO
                if (batch_idx == 3 && row_idx_in_batch < 10)
                    printf("batch %d, row %d, input_row_total %d, input_row_in_batch %d\n", batch_idx, row_idx_in_batch, input_row_idx_total, row_idx_in_batch);
                #endif
                for (int col_idx=0; col_idx < input_width; col_idx+=stride_cols){
                    // sgx_time_t test_start = get_time();
                    #ifdef PRINT_RUN_TIME_INFO
                        im2col_construction_start_time = get_time();
                    #endif
                    const int in_y_origin = row_idx_in_batch - filter_radius_height;
                    const int in_x_origin = col_idx - filter_radius_width;
                    const int out_y = row_idx_in_batch / stride_rows, out_x = col_idx / stride_cols;
                    DtypeForCpuOp* im2col_row_start = im2col_chunk + im2col_row_idx_in_chunk * patch_size;
                    #ifdef PRINT_CONV_IM2COL_CONSTRUCT_INFO
                    if (batch_idx == 3 && out_y==3 and (out_x==2 || out_x==1)){
                        printf(
                            "im2col_row_idx %d:  ", im2col_row_idx_total+im2col_row_idx_in_chunk
                        );
                        printf(
                            "batch %d input_center [%d,%d], origin [%d,%d], out [%d,%d]. ",
                            batch_idx, row_idx_in_batch, col_idx, in_y_origin, in_x_origin, out_y, out_x
                        );
                        printf("(");
                    }
                    #endif
                    for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                        int in_y = in_y_origin + filter_y;
                        #ifdef PRINT_CONV_IM2COL_CONSTRUCT_INFO
                        if (batch_idx == 3 && out_y==3 and (out_x==2 || out_x==1)){
                            printf("in_y-%d", in_y);
                        }
                        #endif
                        if ((in_y < 0) || (in_y >= input_height)) {
                            if (in_y < 0){
                                int start = filter_y*filter_width*input_depth;
                                int end = start + (filter_width * input_depth);
                                #ifdef PRINT_CONV_IM2COL_CONSTRUCT_INFO
                                if (batch_idx == 3 && out_y==3 and out_x==2){
                                    printf("[%d<0,%d-%d]", in_y, start, end);
                                }
                                #endif
                                DtypeForCpuOp* im2col_fill_row_start = im2col_row_start + start;
                                DtypeForCpuOp* im2col_fill_row_end = im2col_row_start + end;
                                    // im2col_fill_row_start + (filter_width * input_depth);
                                std::fill(im2col_fill_row_start, im2col_fill_row_end, DtypeForCpuOp(0));
                            }
                            if (in_y >= input_height){
                                int end = patch_size - (filter_height-1-filter_y)*filter_width*input_depth;
                                int start = end - (filter_width * input_depth);
                                #ifdef PRINT_CONV_IM2COL_CONSTRUCT_INFO
                                if (batch_idx == 3 && out_y==3 && (out_x==2 || out_x==1)){
                                    printf("[%d>=%d,%d-%d]", in_y, input_height, start, end);
                                }
                                #endif
                                DtypeForCpuOp* im2col_fill_row_end =
                                    im2col_row_start + end;
                                DtypeForCpuOp* im2col_fill_row_start = 
                                    im2col_row_start + start;
                                // DtypeForCpuOp* im2col_fill_row_start = 
                                //     im2col_fill_row_end - (filter_width * input_depth);
                                std::fill(im2col_fill_row_start, im2col_fill_row_end, DtypeForCpuOp(0));
                            }
                        } else{
                            const int in_x_end = in_x_origin + filter_width;
                            const int left_zero_count = std::max(0, 0 - in_x_origin);
                            const int right_zero_count = std::max(0, in_x_end - input_width);
                            const int center_copy_count = filter_width - (left_zero_count + right_zero_count);
                            #ifdef PRINT_CONV_IM2COL_CONSTRUCT_INFO
                            if (batch_idx == 3 && out_y==3 && (out_x==2 || out_x==1)){
                                printf("[%d-%d-%d]\n", left_zero_count, center_copy_count, right_zero_count);
                            }
                            #endif
                            if (left_zero_count > 0) {
                                DtypeForCpuOp* im2col_left_start = im2col_row_start + filter_y * filter_width * input_depth;
                                DtypeForCpuOp* im2col_left_end =
                                    im2col_left_start + (left_zero_count * input_depth);
                                std::fill(im2col_left_start, im2col_left_end, DtypeForCpuOp(0));
                                
                            }
                            if (center_copy_count > 0) {
                                int row_bias = filter_y - filter_radius_height;
                                const DtypeForCpuOp* input_row_start =
                                    operate_input_chunk + ((row_idx_in_chunk + row_bias) * input_width * input_depth) +
                                    (std::max(0, in_x_origin) * input_depth);
                                const DtypeForCpuOp* input_row_end =
                                    input_row_start + (center_copy_count * input_depth);
                                DtypeForCpuOp* im2col_center_start =
                                    im2col_row_start + filter_y * filter_width * input_depth + (left_zero_count * input_depth);
                                std::copy(input_row_start, input_row_end, im2col_center_start);
                                #ifdef PRINT_CONV_IM2COL_CONSTRUCT_INFO
                                if (batch_idx == 3 && out_y==3 && (out_x==2 || out_x==1)){
                                    printf("b %d, out_y %d, out_x %d Window %d: ", batch_idx, out_y, out_x, in_y);
                                    for (auto i=0; i<5; i++){
                                        printf("%.2f, ", *(input_row_start+i));
                                    }
                                    printf("\n");
                                }
                                #endif
                            }
                            if (right_zero_count > 0) {
                                DtypeForCpuOp* im2col_right_start =
                                    im2col_row_start + filter_y * filter_width * input_depth +
                                    ((left_zero_count + center_copy_count) * input_depth);
                                DtypeForCpuOp* im2col_right_end =
                                    im2col_right_start + (right_zero_count * input_depth);
                                std::fill(im2col_right_start, im2col_right_end, DtypeForCpuOp(0));
                            }
                        }
                    }
                    #ifdef PRINT_CONV_IM2COL_CONSTRUCT_INFO
                        // printf(") ");
                        // for (int idx=0; idx < patch_size; idx++){
                        //     printf("%.0f, ", *(im2col_row_start+idx));
                        // }
                        // printf("\n");
                    if (batch_idx == 3 && out_y==3 && (out_x==2|| out_x==1)){
                        printf(") \n");
                        for (int idx=0; idx < patch_size; idx++){
                            if (idx%64 == 0)
                                printf("b %d, out_y %d, out_x %d im2col line %d: ", batch_idx, out_y, out_x, idx/64);
                            if (idx%64<5)
                                printf("%.2f, ", *(im2col_row_start+idx));
                            if ((idx+1)%64 == 0)
                                printf("\n");
                        }
                        printf("\n");
                    }
                    #endif
                    im2col_row_idx_in_chunk ++;
                    // output_row_idx_in_chunk ++;
                    #ifdef PRINT_RUN_TIME_INFO
                        im2col_construction_time += get_elapsed_time(im2col_construction_start_time, get_time());
                    #endif

                    // for (int i=0; i<patch_size; i++){
                    //     printf("%f ", im2col_row_start[i]);
                    // }
                    // printf("\n");

                    // bool im2col_last_row = (last_chunk && row_idx_in_chunk == end_row_idx_in_operate_chunk-stride_rows && col_idx == input_width-stride_cols );
                    int last_input_last_row = int((input_height-1)/stride_rows) * stride_rows;
                    int last_col = int((input_width-1)/stride_cols)*stride_cols;
                    bool im2col_last_row = (
                        last_chunk && 
                        input_row_idx_total == (batch-1)*input_height + last_input_last_row && 
                        // row_idx_in_chunk == end_row_idx_in_operate_chunk-stride_rows &&
                        col_idx == last_col
                    );
                    // if (last_chunk)
                    //     printf(
                    //         "last_input_last_row %d, input_row_idx_total %d, row bar %d, col_idx %d, col bar %d \n", 
                    //         last_input_last_row, input_row_idx_total, (batch-1)*input_height + last_input_last_row, col_idx, last_col
                    //     );
                    if ((im2col_row_idx_in_chunk % max_matrix_mul_rows == 0) || 
                        im2col_last_row
                    ){
                        
                        // printf(
                        //     "im2col_row_idx_total %d, im2col_row_idx_in_chunk %d\n", 
                        //     im2col_row_idx_total, im2col_row_idx_in_chunk
                        // );
                        int matrix_mul_rows;
                        if (im2col_row_idx_in_chunk % max_matrix_mul_rows == 0){
                            // assert (output_row_idx_in_chunk % max_matrix_mul_rows == 0);
                            matrix_mul_rows = max_matrix_mul_rows;
                        }
                        else
                            matrix_mul_rows = im2col_row_idx_in_chunk % max_matrix_mul_rows;
                        
                        // set im2col multiply matrix
                        auto im2col_start_row_idx_in_chunk = im2col_row_idx_in_chunk - matrix_mul_rows;
                        auto im2col_mat_start_in_chunk = im2col_chunk + im2col_start_row_idx_in_chunk * patch_size;
                        MapMatRowMajor im2col_mat(im2col_mat_start_in_chunk, matrix_mul_rows, patch_size);

                        // printf("Im2col Mat:\n");
                        // for (auto r=0; r<matrix_mul_rows; r++){
                        //     for (auto c=0; c<patch_size; c++)
                        //         printf("%.0f ", im2col_mat(r,c));
                        //     printf("\n");
                        // }

                        // Not used
                        // auto output_mat_start_in_chunk = output_chunk + (output_row_idx_in_chunk - matrix_mul_rows) * output_c;
                        // printf("Before malloc tmp_output_chunk\n");
                        DtypeForCpuOp* tmp_output_chunk = (DtypeForCpuOp*)malloc(matrix_mul_rows * output_c * sizeof(DtypeForCpuOp));
                        // printf("After malloc tmp_output_chunk, matrix_mul_rows %d, output_c %d\n", matrix_mul_rows, output_c);
                        // printf("Malloc size %d, matrix_mul_rows*output_c %d\n", mspace_usable_size(tmp_output_chunk), matrix_mul_rows*output_c*sizeof(DtypeForCpuOp) );
                        MapMatRowMajor tmp_output_mat(tmp_output_chunk, matrix_mul_rows, output_c);
                        redundant_weight_cnt = 0;
                        int out_channel_start_idx = 0;

                        

                        run_all_chunks([&](int weight_start_store_chunk, int weight_num_elem_in_store_chunk) {
                            int weight_chunk_size_in_byte = weight_num_elem_in_store_chunk * sizeof(DtypeForCpuOp);
                            #ifdef PRINT_RUN_TIME_INFO
                                load_weight_start_time = get_time();
                            #endif
                            chunk_manager.GetChunk(weight_tensor->GetChunkId(weight_start_store_chunk), weight_chunk, weight_chunk_size_in_byte);
                            

                            // printf(
                            //     "Weight_chunk start %d, end %d, num_elem %d, prior redundant %d\n", 
                            //     weight_start_store_chunk, weight_start_store_chunk+weight_num_elem_in_store_chunk, weight_num_elem_in_store_chunk,
                            //     redundant_weight_cnt
                            // );
                            int valid_weight_row_cnt = (redundant_weight_cnt + weight_num_elem_in_store_chunk) / patch_size;
                            // printf("Before malloc operate_weight_chunk\n");
                            operate_weight_chunk = (DtypeForCpuOp*)malloc(valid_weight_row_cnt * patch_size * sizeof(DtypeForCpuOp));
                            // printf("After malloc operate_weight_chunk\n");
                            // printf("Test 1 size %d\n", valid_weight_row_cnt * patch_size);
                            
                            // DtypeForCpuOp* redundent_weight_end_idx = redundent_weight_chunk + redundant_weight_cnt;
                            if (redundant_weight_cnt > 0){
                                // std::copy(redundent_weight_chunk, redundent_weight_end_idx, operate_weight_chunk);
                                // printf("Test 2 redundant size %d, ", redundant_weight_cnt);
                                memcpy(operate_weight_chunk, redundent_weight_chunk, redundant_weight_cnt*sizeof(DtypeForCpuOp));
                                free(redundent_weight_chunk);
                                // printf("Finished\n");
                            }
                            
                            
                            
                            if (redundant_weight_cnt > 0){
                                int copy_to_operate_weight_num_elem = valid_weight_row_cnt * patch_size - redundant_weight_cnt;
                                DtypeForCpuOp* weight_chunk_copy_end = weight_chunk + copy_to_operate_weight_num_elem;
                                DtypeForCpuOp* operate_weight_copy_start = operate_weight_chunk + redundant_weight_cnt;
                                // printf("copy size %d, total size %d\n", copy_to_operate_weight_num_elem, copy_to_operate_weight_num_elem+redundant_weight_cnt);
                                // std::copy(weight_chunk, weight_chunk_copy_end, operate_weight_copy_start);
                                memcpy(operate_weight_copy_start, weight_chunk, copy_to_operate_weight_num_elem*sizeof(DtypeForCpuOp));
                            } else{
                                int copy_to_operate_weight_num_elem = valid_weight_row_cnt * patch_size;
                                memcpy(operate_weight_chunk, weight_chunk, copy_to_operate_weight_num_elem*sizeof(DtypeForCpuOp));
                                // printf("Test 2 size %d\n", copy_to_operate_weight_num_elem);
                            }
                            

                            int new_redundant_weight_cnt = (redundant_weight_cnt + weight_num_elem_in_store_chunk) % patch_size;
                            if (new_redundant_weight_cnt > 0){
                                DtypeForCpuOp* weight_end_in_chunk = weight_chunk + weight_num_elem_in_store_chunk;
                                DtypeForCpuOp* redundent_weight_start_in_chunk = weight_end_in_chunk - new_redundant_weight_cnt;
                                // std::copy(redundent_weight_start_in_chunk, weight_end_in_chunk, redundent_weight_chunk);
                                // printf("Before malloc redundent_weight_chunk\n");
                                redundent_weight_chunk = (DtypeForCpuOp*)malloc(new_redundant_weight_cnt * sizeof(DtypeForCpuOp));
                                // printf("After malloc redundent_weight_chunk\n");
                                memcpy(redundent_weight_chunk, redundent_weight_start_in_chunk, new_redundant_weight_cnt*sizeof(DtypeForCpuOp));
                            }
                            redundant_weight_cnt = new_redundant_weight_cnt;

                            int out_channels_in_chunk = valid_weight_row_cnt;
                            // int out_channel_start_idx = weight_start_store_chunk / patch_size;
                            // printf("Tag2\n");
                            MapMatRowMajor weight_mat(operate_weight_chunk, out_channels_in_chunk, patch_size);
                            // printf("Weight mat\n");
                            
                            // printf(
                            //     "Weight rows %d, after redundant %d \n",
                            //     valid_weight_row_cnt, redundant_weight_cnt
                            // );

                            // printf("Weight\n");
                            // for (auto r=0; r<out_channels_in_chunk; r++){
                            //     for (auto c=0; c<patch_size; c++){
                            //         printf("%.0f ", weight_mat(r,c));
                            //     }
                            //     printf("\n");
                            // }
                            #ifdef PRINT_RUN_TIME_INFO
                                load_weight_time += get_elapsed_time(load_weight_start_time, get_time());
                            #endif
                            #ifdef PRINT_RUN_TIME_INFO
                                matrix_mul_start_time = get_time();
                            #endif
                            int tmp_output_row_start;
                            if (max_output_patches_per_chunk == max_matrix_mul_rows)
                                tmp_output_row_start = 0;
                            else
                                tmp_output_row_start = im2col_start_row_idx_in_chunk;
                            // printf(
                            //     "Block info: tmp_output_row_start %d, out_channel_start_idx %d, matrix_mul_rows %d, out_channels_in_chunk %d\n", 
                            //     tmp_output_row_start, out_channel_start_idx, matrix_mul_rows, out_channels_in_chunk
                            // );
                            auto output_block = tmp_output_mat.block(
                                tmp_output_row_start, out_channel_start_idx, matrix_mul_rows, out_channels_in_chunk
                            );
                            // printf("Block\n");
                            output_block.array() = im2col_mat * weight_mat.transpose();
                            #ifdef PRINT_RUN_TIME_INFO
                                matrix_mul_time += get_elapsed_time(matrix_mul_start_time, get_time());
                            #endif
                            // printf("Multiply\n");

                            out_channel_start_idx += out_channels_in_chunk;
                            free(operate_weight_chunk);
                            // printf("Free\n");
                        }, STORE_CHUNK_ELEM, weight_tensor->GetNumElem());
                        if (redundant_weight_cnt > 0){
                            printf("redundant_weight_cnt is not 0, is %d\n", redundant_weight_cnt);
                        }
                        assert (redundant_weight_cnt == 0);


                        // add bias
                        // printf("Bias: \n");
                        // for (auto i=0; i<output_c; i++)
                        //     printf("%f ", bias_mat(0,i));
                        // printf("\n");
                        for (auto i=0; i<matrix_mul_rows; i++){
                            auto output_block = tmp_output_mat.block(i, 0, 1, output_c);
                            output_block.array() = output_block + bias_mat;
                        }


                        // printf("Output: %d\n", matrix_mul_rows);
                        // for (int r=0; r<matrix_mul_rows; r++){
                        //     for (int c=0; c<1; c++){
                        //         printf("%f ", tmp_output_mat(r,c));
                        //     }
                            
                        // }
                        // printf("\n");

                        if ((output_row_idx_in_chunk + matrix_mul_rows <= output_patches_per_chunk) ||
                            im2col_last_row
                        ){
                            #ifdef PRINT_RUN_TIME_INFO
                                save_output_start_time = get_time();
                            #endif
                            #ifdef PRINT_CONV_OUTPUT_SAVE_CHUNK_INFO
                                printf(
                                    "Directly copy, im2col_last_row %d, ", im2col_last_row);
                                printf(
                                    "im2col_last_row %d, last_chunk %d, row_idx_in_chunk %d (%d-%d), output_row_idx_in_chunk %d, total output row %d-%d\n",
                                    im2col_last_row, last_chunk, row_idx_in_chunk, end_row_idx_in_operate_chunk, stride_rows,
                                    output_row_idx_in_chunk, 
                                    output_row_idx_total+output_row_idx_in_chunk, output_row_idx_total+output_row_idx_in_chunk+matrix_mul_rows
                                );
                                // for (auto print_output_row_idx=0; print_output_row_idx<matrix_mul_rows; print_output_row_idx++){
                                //     if (output_row_idx_total+output_row_idx_in_chunk+print_output_row_idx == 2438){
                                //         DtypeForCpuOp* p_print = tmp_output_chunk+print_output_row_idx;
                                //         printf("Row %d output %.5f\n",output_row_idx_total+output_row_idx_in_chunk+print_output_row_idx, *p_print);
                                //     }
                                // }
                            #endif
                            // directly copy
                            DtypeForCpuOp* output_start_idx_in_chunk = output_chunk + (output_row_idx_in_chunk * output_c);
                            DtypeForCpuOp* tmp_output_chunk_end = tmp_output_chunk + (matrix_mul_rows * output_c);
                            // std::copy(tmp_output_chunk, tmp_output_chunk_end, output_start_idx_in_chunk);
                            memcpy(output_start_idx_in_chunk, tmp_output_chunk, matrix_mul_rows*output_c*sizeof(DtypeForCpuOp));
                            output_row_idx_in_chunk += matrix_mul_rows;
    
                            if (im2col_last_row || output_row_idx_in_chunk == output_patches_per_chunk){
                                #ifdef PRINT_CONV_OUTPUT_SAVE_CHUNK_INFO
                                    printf(
                                        "Last Store, last_row %d, output rows %d/%d\n", 
                                        im2col_last_row, output_row_idx_in_chunk, output_patches_per_chunk
                                    );
                                    
                                #endif
                                // printf("output_row_idx_total(%d)*output_c(%d) = %d\n", output_row_idx_total, output_c, output_row_idx_total*output_c);
                                // save output_chunk
                                chunk_manager.StoreChunk(
                                    output_tensor->GetChunkId(output_row_idx_total*output_c), output_chunk, 
                                    output_row_idx_in_chunk*output_c*sizeof(DtypeForCpuOp)
                                );
                                output_row_idx_total += output_row_idx_in_chunk;
                                output_row_idx_in_chunk = 0;
                            }
                            #ifdef PRINT_RUN_TIME_INFO
                                save_output_time += get_elapsed_time(save_output_start_time, get_time());
                            #endif
                        } else{
                            #ifdef PRINT_RUN_TIME_INFO
                                save_output_start_time = get_time();
                            #endif
                            #ifdef PRINT_CONV_OUTPUT_SAVE_CHUNK_INFO
                                printf(
                                    "Middle copy, old output rows %d -> %d id %ld, ", 
                                    output_row_idx_in_chunk, output_patches_per_chunk, 
                                    output_tensor->GetChunkId(output_row_idx_total*output_c)
                                );
                            #endif
                            //copy part of tmp_output, store output_chunk, load new output_chunk, and copy the rest
                            DtypeForCpuOp* output_start_idx_in_chunk = output_chunk + (output_row_idx_in_chunk * output_c);
                            int chunk_left_rows = output_patches_per_chunk - output_row_idx_in_chunk;
                            DtypeForCpuOp* tmp_output_chunk_divide = tmp_output_chunk + (chunk_left_rows * output_c);
                            std::copy(tmp_output_chunk, tmp_output_chunk_divide, output_start_idx_in_chunk);
                            output_row_idx_in_chunk += chunk_left_rows;
                            chunk_manager.StoreChunk(
                                output_tensor->GetChunkId(output_row_idx_total*output_c), output_chunk, 
                                output_row_idx_in_chunk*output_c*sizeof(DtypeForCpuOp)
                            );

                            while (matrix_mul_rows-chunk_left_rows > output_patches_per_chunk){
                                DtypeForCpuOp* tmp_output_chunk_prev_divide = tmp_output_chunk_divide;
                                output_row_idx_total += output_patches_per_chunk;
                                output_start_idx_in_chunk = output_chunk;
                                tmp_output_chunk_divide += output_patches_per_chunk * output_c;
                                std::copy(tmp_output_chunk_prev_divide, tmp_output_chunk_divide, output_start_idx_in_chunk);
                                chunk_manager.StoreChunk(
                                    output_tensor->GetChunkId(output_row_idx_total*output_c), output_chunk, 
                                    output_patches_per_chunk*output_c*sizeof(DtypeForCpuOp)
                                );
                                chunk_left_rows += output_patches_per_chunk;
                                #ifdef PRINT_CONV_OUTPUT_SAVE_CHUNK_INFO
                                    printf(
                                        "middle integral output 0 -> %d id %ld, ",
                                        output_patches_per_chunk, output_tensor->GetChunkId(output_row_idx_total*output_c)
                                    );
                                #endif
                            }

                            output_row_idx_total += output_patches_per_chunk;
                            output_row_idx_in_chunk = 0;
                            output_start_idx_in_chunk = output_chunk + (output_row_idx_in_chunk * output_c);
                            DtypeForCpuOp* tmp_output_chunk_end = tmp_output_chunk + (matrix_mul_rows * output_c);
                            std::copy(tmp_output_chunk_divide, tmp_output_chunk_end, output_start_idx_in_chunk);
                            output_row_idx_in_chunk += (matrix_mul_rows - chunk_left_rows);
                            #ifdef PRINT_CONV_OUTPUT_SAVE_CHUNK_INFO
                                printf(
                                    "new output rows 0 -> %d id %ld\n",
                                    output_row_idx_in_chunk, output_tensor->GetChunkId(output_row_idx_total*output_c)
                                );
                            #endif
                            #ifdef PRINT_RUN_TIME_INFO
                                save_output_time += get_elapsed_time(save_output_start_time, get_time());
                            #endif
                        }

                        // Reset im2col_row_idx_in_chunk
                        if (im2col_row_idx_in_chunk + matrix_mul_rows >= im2col_patches_per_chunk){
                            im2col_row_idx_total += im2col_row_idx_in_chunk;
                            im2col_row_idx_in_chunk = 0;
                        }
                        free(tmp_output_chunk);
                    }
                    // test_time += get_elapsed_time(test_start, get_time());
                }
                // Hard coding for stride = 1, 2
                if (row_idx_in_batch + stride >= input_height){
                    input_row_idx_total += input_height - row_idx_in_batch;
                    row_idx_in_chunk += input_height - row_idx_in_batch;
                }
                else{
                    input_row_idx_total += stride_rows;
                    row_idx_in_chunk += stride_rows;
                }
            
            }
            free(operate_input_chunk);
            
        }, input_elem_fetch_per_chunk, input_tensor->GetNumElem());
    #ifdef PRINT_RUN_TIME_INFO
        printf(
            "Load input time %lf, im2col time %lf, matrix mul time %lf, save output time %lf, load weight time %lf, forward prepare time %lf\n", 
            load_input_time, im2col_construction_time, matrix_mul_time, save_output_time, load_weight_time, forward_prepare_time );
    #endif

    #ifdef PRINT_RUN_TIME_INFO
        total_time += get_elapsed_time(total_start, get_time());
        printf("Total time %lf\n", total_time);
        printf("Test time %lf\n", test_time);
    #endif
    }

    void backward() {}
    
    IdT FunId;
    int batch, input_h, input_w, input_c;
    int output_h, output_w, output_c, kernel, padding, stride;

    int input_row_size, one_batch_input_size, input_elem_fetch_per_chunk, 
        patch_size, max_im2col_patches_per_chunk, max_output_patches_per_chunk, 
        max_weight_rows_per_chunk, max_weight_elem_per_chunk,
        max_matrix_mul_rows, im2col_num_elem_in_chunk, output_num_elem_in_chunk,
        im2col_patches_per_chunk, output_patches_per_chunk;

    int num_rows;
    int num_rows_in_channel;
    int total_n;
    int default_num_batches_per_chunk, default_num_col_per_chunk;
    int features_per_chunk, classes_per_chunk;

    int NumBatchesTrackedArr = 0;

    #ifdef PRINT_RUN_TIME_INFO
        sgx_time_t load_input_time=0, load_weight_time=0, im2col_construction_time=0, 
        save_output_time=0, matrix_mul_time=0, forward_prepare_time=0, total_time=0, test_time=0;
    #endif

    shared_ptr<SecretTen> input_tensor;
    shared_ptr<SecretTen> output_tensor;
    // shared_ptr<SecretTen> der_input_tensor;
    // shared_ptr<SecretTen> der_output_tensor;
    shared_ptr<SecretTen> weight_tensor;
    shared_ptr<SecretTen> bias_tensor;
    // shared_ptr<SecretTen> der_weight_tensor;
    // shared_ptr<SecretTen> der_bias_tensor;
};


class MaxpoolBuffer {
public:
    MaxpoolBuffer() {}
    MaxpoolBuffer(IdT FunId_, IdT TenIdin_trans_, IdT TenIdout_trans_) : FunId(FunId_), TenIdin_trans(TenIdin_trans_), TenIdout_trans(TenIdout_trans_)  { }

    ~MaxpoolBuffer() = default;

	IdT get_TenIdin_trans(){
		return TenIdin_trans;
	}

	IdT get_TenIdout_trans(){
		return TenIdout_trans;
	}
    //if NCHW->WHCN N=CN M=HW
    void transpose(const DtypeForCpuOp *src, DtypeForCpuOp *dst, const size_t N, const size_t M) {
    #pragma omp parallel for
       for(size_t n = 0; n<N*M; n++) {
          size_t i = n/N;
          size_t j = n%N;
          dst[n] = src[M*j + i]; 
       }   
    }

    inline void transpose4x4_SSE(const float *A, float *B, const uint32_t lda, const uint32_t ldb) {
        __m128 row1 = _mm_load_ps(&A[0*lda]);
        __m128 row2 = _mm_load_ps(&A[1*lda]);
        __m128 row3 = _mm_load_ps(&A[2*lda]);
        __m128 row4 = _mm_load_ps(&A[3*lda]);
         _MM_TRANSPOSE4_PS(row1, row2, row3, row4);
         _mm_store_ps(&B[0*ldb], row1);
         _mm_store_ps(&B[1*ldb], row2);
         _mm_store_ps(&B[2*ldb], row3);
         _mm_store_ps(&B[3*ldb], row4);
    }

    inline void transpose_block_SSE4x4(const float *A, float *B, const uint32_t lda, const uint32_t ldb ,const int block_size) {
        #pragma omp parallel for
        for(uint32_t i=0; i<ldb; i+=block_size) {
            for(uint32_t j=0; j<lda; j+=block_size) {
                uint32_t max_i2 = i+block_size < ldb ? i + block_size : ldb;
                uint32_t max_j2 = j+block_size < lda ? j + block_size : lda;
                for(uint32_t i2=i; i2<max_i2; i2+=4) {
                    for(uint32_t j2=j; j2<max_j2; j2+=4) {
                        transpose4x4_SSE(&A[i2*lda +j2], &B[j2*ldb + i2], lda, ldb);
                    }
                }
            }
         }
    }
    
    inline void MaxpoolAVX(const uint32_t num_img, float* input, float* output){
        // uint32_t base = num_img/8;
        // base *= 8;
        // printf("Base %d\n", base);

        // AVX requires to align address to 32 bytes, both input and output should align
        // printf("mod 8 %d\n", input)
        // #pragma omp parallel for        
        // for(size_t i=0; i<base; i+=8){
        //     printf("%d, addr %p\n", i, input);
        //     const __m256 inp8f = _mm256_load_ps(&input[i]);
        //     const __m256 out8f = _mm256_load_ps(&output[i]);
        //     const __m256 if_lq = _mm256_cmp_ps(out8f, inp8f, 0x01);
        //     const __m256 res8f = _mm256_blendv_ps(out8f, inp8f, if_lq);
        //     _mm256_stream_ps(&output[i], res8f);
        // }
        // printf("Middle\n");

        for (size_t i=0; i<num_img; i++){
            if (input[i] > output[i])
                output[i] = input[i];
        }
        // printf("Finish\n");
    }

    inline void PlainMaxpool(const uint32_t num_img, float* input, float* output){
        #pragma omp parallel for        
        for (size_t i=0; i<num_img; i++){
            if (input[i] > output[i])
                output[i] = input[i];
        }
    }

    inline void MaxpoolbackAVX(const uint32_t num_img, float* input, float* output, float* dinput, float* doutput){
        #pragma omp parallel for
        for(size_t i=0; i<num_img; i+=8){
            const __m256 inp8f = _mm256_load_ps(&input[i]);
            const __m256 out8f = _mm256_load_ps(&output[i]);
            const __m256 din8f = _mm256_load_ps(&dinput[i]);
            const __m256 dout8f = _mm256_load_ps(&doutput[i]);
            const __m256 if_eq = _mm256_cmp_ps(out8f, inp8f, 0x00);
            const __m256 sum8f = _mm256_add_ps(din8f, dout8f);
            const __m256 res8f = _mm256_blendv_ps(din8f, sum8f, if_eq); // define dinput
            const __m256 res28f = _mm256_blendv_ps(dout8f, zero8f, if_eq); // redefine doutput
            _mm256_store_ps(&dinput[i], res8f);
            _mm256_stream_ps(&doutput[i], res28f);
        }
    }

    void forward(
           shared_ptr<SecretTen> ten_in, shared_ptr<SecretTen> ten_out,
           shared_ptr<SecretTen> ten_in_trans, shared_ptr<SecretTen> ten_out_trans,
           uint32_t batch, uint32_t channel,uint32_t input_height, uint32_t input_width,
           uint32_t output_height, uint32_t output_width, uint32_t filter_height,
           uint32_t filter_width, uint32_t row_stride, uint32_t col_stride) {

        const uint32_t inputhw = input_height*input_width;
        uint32_t num_img_in_storechunk = STORE_CHUNK_ELEM/inputhw;

		if(STORE_CHUNK_ELEM % inputhw != 0){
			printf("!!!!!!!!!!!!!!!!!!! STORE_CHUNK_ELEM %% inputhw != 0 !!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
			return;
		}
        if (channel % 8 != 0){
            printf("Channel (%d) % 8 should be 0, but channel is not!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
            printf("Please change the form of AVX functions\n");
            return;
        }
		//if (num_img_in_storechunk % 8 != 0){
		//	printf("STORE_CHUNK_ELEM/inputhw is not divisible by 8!\n");
		//	return;
		//}

        const uint32_t radius_height = filter_height/2, radius_width = filter_width/2;
        const uint32_t outputhw = output_height * output_width;
        uint32_t outputsize_in_storechunk = num_img_in_storechunk * outputhw;
        const uint32_t total_size = batch * channel * inputhw;
        size_t idx_out=0;
        size_t idx_tmp=0;
		size_t size_of_store_chunk = STORE_CHUNK_ELEM * sizeof(float);      
        bool if_use_SSE_out =(outputhw%4==0);

        float* chunk_in, *chunk_out, *chunk_in_trans, *chunk_out_trans, *chunk_tmp;
		auto& chunk_manager = TrustedChunkManager::getInstance();

        ChunkGuard<DtypeForCpuOp> guard_in(StoreChunkPool::GetChunkPool(), chunk_in);
        ChunkGuard<DtypeForCpuOp> guard_out(StoreChunkPool::GetChunkPool(), chunk_out);
        ChunkGuard<DtypeForCpuOp> guard_int(StoreChunkPool::GetChunkPool(), chunk_in_trans);
        ChunkGuard<DtypeForCpuOp> guard_outt(StoreChunkPool::GetChunkPool(), chunk_out_trans);
        ChunkGuard<DtypeForCpuOp> guard_tmp(StoreChunkPool::GetChunkPool(), chunk_tmp); // chunk_tmp is used to store output temporarily

        auto chunk_op = [&](size_t start_chunk, size_t num_elem_in, size_t num_elem_out) {
            num_img_in_storechunk = num_elem_in / inputhw;
            size_of_store_chunk = num_elem_in * sizeof(float);
            // printf("maxpooling forward in enclave. start_chunk: %d, num_elem_in %d, num_elem_out %d\n", start_chunk, num_elem_in, num_elem_out);
            chunk_manager.GetChunk(ten_in->GetChunkId(start_chunk), chunk_in, num_elem_in * sizeof(DtypeForCpuOp));
            // printf("Input: h=%d, w=%d\n", input_height, input_width);
            // for (auto ii=0; ii<input_height; ii++){
            //     for (auto jj=0; jj<input_width; jj++){
            //         printf("%f ", *(chunk_in+ii*input_width+jj));
            //     }
            //     printf("\n");
            // }
            // printf("transpose inputhw %d, num_img_in_storechunk %d \n", inputhw, num_img_in_storechunk);
            transpose_block_SSE4x4(chunk_in, chunk_in_trans, inputhw, num_img_in_storechunk, 8);
            // transpose(chunk_in, chunk_in_trans, num_img_in_storechunk, inputhw);
            // Save transpose chunk have problem when STORE_CHUNK_ELEM=1204224, channel=1024, imghw=8
            // chunk_manager.StoreChunk(ten_in_trans->GetChunkId(start_chunk), chunk_in_trans, size_of_store_chunk);
            // printf("Transpose input: h=%d, w=%d\n", input_height, input_width);
            // for (auto ii=0; ii<input_height; ii++){
            //     for (auto jj=0; jj<input_width; jj++){
            //         printf("%f ", *(chunk_in_trans+ii*input_width+jj));
            //     }
            //     printf("\n");
            // }
			fill(chunk_out_trans, chunk_out_trans + outputsize_in_storechunk, std::numeric_limits<DtypeForCpuOp>::lowest());
            for(uint32_t h = 0; h < input_height; ++h) {
                for(uint32_t w = 0; w < input_width; ++w) {
                    // (h_start, h_end) * (w_start, w_end) is the range that the input
                    // vector projects to.
                    // const uint32_t h_start = (h < filter_height)
                    //                         ? 0
                    //                         : (h - filter_height) / row_stride + 1;
                    // const uint32_t h_end = std::min(h / row_stride + 1, output_height);
                    // const uint32_t w_start = (w < filter_width)
                    //                         ? 0
                    //                         : (w - filter_width) / col_stride + 1;
                    // const uint32_t w_end = std::min(w / col_stride + 1, output_width);


                    // const uint32_t h_start = h / row_stride;
                    // const uint32_t h_end = std::min((h+filter_height)/row_stride, output_height);
                    // const uint32_t w_start = w / col_stride;
                    // const uint32_t w_end = std::min((w+filter_width)/col_stride , output_width);

                    uint32_t h_start = (h < radius_height)
                                        ? 0
                                        : (h-radius_height + row_stride-1)/row_stride;
                    uint32_t h_end = (h+radius_height)/row_stride+1;
                    h_end = std::min<uint32_t>(h_end, output_height);

                    uint32_t w_start = (w < radius_width)
                                        ? 0
                                        : (w-radius_width + col_stride-1)/col_stride;
                    uint32_t w_end = (w+radius_width)/col_stride+1;
                    w_end = std::min<uint32_t>(w_end, output_width);

                    // if (h==0 && w==0)
                    // printf(
                    //     "(%d, %d): h[%d, %d], w[%d, %d]\n",
                    //     h, w, h_start, h_end, w_start, w_end
                    // );
                    // compute elementwise max
                    const uint32_t in_offset = (h * input_width + w)*num_img_in_storechunk;
                    for (uint32_t ph = h_start; ph < h_end; ++ph) {
                        const uint32_t out_offset_base = ph * output_width;
                        for (uint32_t pw = w_start; pw < w_end; ++pw) {
                            const uint32_t out_offset = (out_offset_base + pw) * num_img_in_storechunk;
                            // printf(
                            //     "ph %d, pw %d, in_offset %d, out_offset_base %d, out_offset %d\n",
                            //     ph, pw, in_offset, out_offset_base, out_offset
                            // );
                            
                            // MaxpoolAVX(num_img_in_storechunk, chunk_in_trans+in_offset, chunk_out_trans + out_offset);
                            PlainMaxpool(num_img_in_storechunk, chunk_in_trans+in_offset, chunk_out_trans + out_offset);
                        }
                    }
                }
            }
            // chunk_manager.StoreChunk(ten_out_trans->GetChunkId(start_chunk), chunk_out_trans, size_of_store_chunk);
            // printf("Save transposed output\n");
            // printf("Transpose output: h=%d, w=%d\n", output_height, output_width);
            // for (auto ii=0; ii<output_height; ii++){
            //     for (auto jj=0; jj<output_width; jj++){
            //         printf("%f ", *(chunk_out_trans+ii*output_width+jj));
            //     }
            //     printf("\n");
            // }
            // printf("use SSE %d\n", if_use_SSE_out);
            //transpose
            if(if_use_SSE_out){
                transpose_block_SSE4x4(chunk_out_trans, chunk_tmp, num_img_in_storechunk, outputhw, 8);
            }
            else{
                transpose(chunk_out_trans, chunk_tmp, outputhw, num_img_in_storechunk);
            }
            // transpose(chunk_out_trans, chunk_tmp, outputhw, num_img_in_storechunk);
            if(idx_tmp+num_elem_out<STORE_CHUNK_ELEM){
                copy(chunk_tmp, chunk_tmp+num_elem_out, chunk_out + idx_tmp);
                idx_tmp+=num_elem_out;
            }
            else{
                size_t idx_add = STORE_CHUNK_ELEM-idx_tmp;
                copy(chunk_tmp,chunk_tmp+idx_add,chunk_out+idx_tmp);
                chunk_manager.StoreChunk(ten_out->GetChunkId(idx_out), chunk_out, size_of_store_chunk);
                idx_out += STORE_CHUNK_ELEM;
                copy(chunk_tmp + idx_add,chunk_tmp + num_elem_out,chunk_out + idx_tmp+idx_add);
                idx_tmp += num_elem_out;
				idx_tmp -= STORE_CHUNK_ELEM; 
            }
        };//end of chunk_op
        run_all_chunks_for_maxpool(chunk_op, STORE_CHUNK_ELEM, batch * channel * inputhw, outputsize_in_storechunk, inputhw, outputhw);      

        if (idx_tmp!=0) {
            chunk_manager.StoreChunk(ten_out->GetChunkId(idx_out), chunk_out, idx_tmp * sizeof(DtypeForCpuOp)); 
        }
	}//end maxpooling

    
    void backward(
            shared_ptr<SecretTen> ten_din, shared_ptr<SecretTen> ten_dout,
            shared_ptr<SecretTen> ten_in_trans, shared_ptr<SecretTen> ten_out_trans,
            uint32_t batch, uint32_t channel,uint32_t input_height, uint32_t input_width,
            uint32_t output_height, uint32_t output_width,
            uint32_t filter_height, uint32_t filter_width, uint32_t row_stride, uint32_t col_stride) {
	}//end maxpoolbackward


    IdT FunId;
	IdT TenIdin_trans;
   	IdT TenIdout_trans;
};

static inline float float2_to_uniform(uint32_t x, uint32_t y, float& a, float& b) {
    const union { uint32_t i; float d;  } u = { .i = UINT32_C(0x7F) << 23 | ((x ^ y) >> 2) };
    const union { uint32_t i; float d;  } v = { .i = UINT32_C(0x7F) << 23 | (((x ^ y) >> 5) ^ UINT32_C(0x7FFFFF))};
    a = u.d - 1.0f;
    b = v.d - 1.0f;
}

extern "C" {

void SecretInitQuantTensor(IdT TenId, void *voidDims) {
    DimsT dims = *(DimsT*)voidDims;
    // #ifdef PRINT_CHUNK_INFO
    // printf("SecretInitQuantTensor id %llu, size (%d,%d,%d,%d), \n", TenId, dims.dim0, dims.dim1, dims.dim2, dims.dim3);
    // #endif
    DimsT *Dims = (DimsT *) voidDims;
    SecretQuantTenHolder[TenId] = make_shared<SecretQuantTen>(TenId, Dims);
}

void SecretSetQuantTen(IdT TenId, void *voidArr) {
    DtypeForQuant* cpu_p = (DtypeForQuant*) voidArr;
    // printf("TenId %ld, %f %f %f\n", TenId, cpu_p[0], cpu_p[1], cpu_p[2]);
    GetQuantTenById(TenId)->SetTen((DtypeForQuant *) voidArr);
}

void SecretGetQuantTen(IdT TenId, void *voidArr) {
    GetQuantTenById(TenId)->GetTen((DtypeForQuant *) voidArr);
}

void SecretInitTensor(IdT TenId, void *voidDims) {
    DimsT dims = *(DimsT*)voidDims;
    #ifdef PRINT_CHUNK_INFO
        printf("SecretInitTensor id %ld, size (%d,%d,%d,%d), ", TenId, dims.dim0, dims.dim1, dims.dim2, dims.dim3);
    #endif
    DimsT *Dims = (DimsT *) voidDims;
    SecretTenHolder[TenId] = make_shared<SecretTen>(TenId, Dims);
}

void SecretSetTen(IdT TenId, void *voidArr) {
    DtypeForCpuOp* cpu_p = (DtypeForCpuOp*) voidArr;
    // printf("TenId %ld, %f %f %f\n", TenId, cpu_p[0], cpu_p[1], cpu_p[2]);
    GetTenById(TenId)->SetTen((DtypeForCpuOp *) voidArr);
}

void SecretGetTen(IdT TenId, void *voidArr) {
    GetTenById(TenId)->GetTen((DtypeForCpuOp *) voidArr);
}

void SecretSetSeed(IdT TenId, uint64_t RawSeed) {
    GetTenById(TenId)->SetSeed(RawSeed);
}

void SecretAddFromCpu(void* inputArr, IdT dstId) {
    shared_ptr<SecretTen > StoreTensor = GetTenById(dstId);
    DtypeForCpuOp PLimit = static_cast<DtypeForCpuOp>(PrimeLimit);
    DtypeForCpuOp invPLimit = static_cast<double>(1) / PrimeLimit;

    const int total_num_elem = StoreTensor->GetNumElem();
    auto& chunk_manager = TrustedChunkManager::getInstance();
    DtypeForCpuOp* store_chunk;
    ChunkGuard<DtypeForCpuOp> guard(StoreChunkPool::GetChunkPool(), store_chunk);

    auto store_chunk_op = [&](int start_store_chunk, int num_elem_in_store_chunk) {
        chunk_manager.GetChunk(StoreTensor->GetChunkId(start_store_chunk), store_chunk, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));

        auto chunk_op = [&](int start_chunk, int num_elem_in_op) {
            DtypeForCpuOp* output_arr = store_chunk + start_chunk;
            DtypeForCpuOp* input_arr = ((DtypeForCpuOp*) inputArr) + start_store_chunk + start_chunk;
            for(size_t j = 0; j < num_elem_in_op; j++) {
                output_arr[j] += input_arr[j];
                output_arr[j] -= floor(output_arr[j] * invPLimit) * PLimit;
                output_arr[j] = (output_arr[j] >= mid) ? (output_arr[j] - p) : output_arr[j];
            }
        };
        run_all_chunks(chunk_op, WORK_CHUNK_ELEM, num_elem_in_store_chunk);

        chunk_manager.StoreChunk(StoreTensor->GetChunkId(start_store_chunk), store_chunk, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));
    };
    run_all_chunks(store_chunk_op, STORE_CHUNK_ELEM, total_num_elem);
}

void newrelu(IdT TenIdin, IdT TenIdout, uint64_t size){
    shared_ptr<SecretTen > ten_in = GetTenById(TenIdin);
	shared_ptr<SecretTen > ten_out = GetTenById(TenIdout);
	auto& chunk_manager = TrustedChunkManager::getInstance();
    DtypeForCpuOp* chunk_in,* chunk_tmp;
    ChunkGuard<DtypeForCpuOp> guard_tmp(StoreChunkPool::GetChunkPool(), chunk_tmp);
    // printf("Newrelu\n");
    //ChunkGuard<DtypeForCpuOp> guard_out(StoreChunkPool::GetChunkPool(), chunk_out);
    auto store_chunk_op = [&](int start_store_chunk, int num_elem_in_store_chunk) {
        chunk_manager.GetChunk(ten_in->GetChunkId(start_store_chunk), chunk_tmp, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));
		// for(uint64_t i=0;i<num_elem_in_store_chunk;i+=8){
        //     const __m256 inp8f = _mm256_load_ps(&chunk_tmp[i]);         
        //     const __m256 if_gt = _mm256_cmp_ps(inp8f, zero8f, 0x0e);
        //     const __m256 res8f = _mm256_blendv_ps(zero8f, inp8f, if_gt);
        //     _mm256_stream_ps(&chunk_tmp[i], res8f);
        // } 
        for (int i=0; i<num_elem_in_store_chunk; i++){
            chunk_tmp[i] = chunk_tmp[i]>0 ? chunk_tmp[i]: 0;
        }
		chunk_manager.StoreChunk(ten_out->GetChunkId(start_store_chunk), chunk_tmp, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));
    };
    run_all_chunks(store_chunk_op, STORE_CHUNK_ELEM, size);
}


void quantrelu(IdT TenIdin, IdT TenIdout, uint64_t size, int exp, int bits){
    shared_ptr<SecretQuantTen > ten_in = GetQuantTenById(TenIdin);
	shared_ptr<SecretQuantTen > ten_out = GetQuantTenById(TenIdout);
	auto& chunk_manager = TrustedChunkManagerUint8::getInstance();
    DtypeForQuant* chunk_in,* chunk_tmp;
    DtypeForCpuOp *chunk_float, *chunk_float_input;
    // printf("Before chunk_tmp %x\n", chunk_tmp);
    ChunkGuard<DtypeForQuant> guard_tmp(StoreChunkPool::GetChunkPool(), chunk_tmp);
    ChunkGuard<DtypeForCpuOp> guard_float(StoreChunkPool::GetChunkPool(), chunk_float);
    ChunkGuard<DtypeForCpuOp> guard_float_input(StoreChunkPool::GetChunkPool(), chunk_float_input);
    // printf("After chunk_tmp %x\n", chunk_tmp);
    // printf("Quant relu, size %d, exp %d, bits %d\n", size, exp, bits);
    //ChunkGuard<DtypeForCpuOp> guard_out(StoreChunkPool::GetChunkPool(), chunk_out);
    auto store_chunk_op = [&](int start_store_chunk, int num_elem_in_store_chunk) {
        // printf("num_elem_in_store_chunk %d * %d, start_store_chunk %d \n", num_elem_in_store_chunk, sizeof(DtypeForQuant), start_store_chunk);
        chunk_manager.GetChunk(ten_in->GetChunkId(start_store_chunk), chunk_tmp, num_elem_in_store_chunk * sizeof(DtypeForQuant));
        // printf("Quant input\n");
        // for (int i=0; i<16; i++){
        //     for (int j=0; j<4; j++)
        //         printf("%u, ", chunk_tmp[i*4+j]);
        //     printf("\n");
        // }
        DtypeForCpuOp shrink_factor = pow(2, exp - (bits - 2) );
        DtypeForCpuOp bias = pow(2, bits-1);
        for (int i=0; i<num_elem_in_store_chunk; i++){
            chunk_float[i] = ((float)chunk_tmp[i] - bias) * shrink_factor;
            chunk_float[i] = chunk_float[i]>0 ? chunk_float[i] : 0;
            chunk_tmp[i] = (DtypeForQuant) (chunk_float[i] / shrink_factor + bias);
        }
            
        // printf("Quant float output\n");
        // for (int i=0; i<16; i++){
        //     for (int j=0; j<4; j++)
        //         printf("%u, ", chunk_tmp[i*4+j]);
        //     printf("\n");
        // }

        // MapEigenTensor src_map = MapEigenTensor(chunk_float_input, 1, 1, 1, num_elem_in_store_chunk);
        // MapEigenTensor dst_map = MapEigenTensor(chunk_float, 1, 1, 1, num_elem_in_store_chunk);

        // DtypeForCpuOp shrink_factor = pow(2, exp - (bits - 2) );
        // dst_map = (src_map - pow(2, bits-1) ) * shrink_factor;

        // printf("Float input\n");
        // for (int i=0; i<16; i++){
        //     for (int j=0; j<4; j++)
        //         printf("%.2f, ", chunk_float[i*4+j]);
        //     printf("\n");
        // }
        
        chunk_manager.StoreChunk(ten_out->GetChunkId(start_store_chunk), chunk_tmp, num_elem_in_store_chunk * sizeof(DtypeForQuant));
        // printf("After StoreChunk\n");


		// for(uint64_t i=0;i<num_elem_in_store_chunk;i+=8){
        //     const __m256 inp8f = _mm256_load_ps(&chunk_tmp[i]);         
        //     const __m256 if_gt = _mm256_cmp_ps(inp8f, zero8f, 0x0e);
        //     const __m256 res8f = _mm256_blendv_ps(zero8f, inp8f, if_gt);
        //     _mm256_stream_ps(&chunk_tmp[i], res8f);
        // } 
		// chunk_manager.StoreChunk(ten_out->GetChunkId(start_store_chunk), chunk_tmp, num_elem_in_store_chunk * sizeof(DtypeForQuant));
    };
    run_all_chunks(store_chunk_op, STORE_CHUNK_ELEM, size);
}

unordered_map<IdT, shared_ptr<MaxpoolBuffer>> MaxpoolHolder;


shared_ptr<MaxpoolBuffer> GetBufferByIdM(IdT FunId) {
    return MaxpoolHolder[FunId];
}

void initmaxpool(IdT FunId, IdT TenIdin_trans, IdT TenIdout_trans){	
    // printf("Initmaxpool TenIdin_trans %ld, TenIdout_trans %ld\n", TenIdin_trans, TenIdout_trans);
    MaxpoolHolder[FunId] = make_shared<MaxpoolBuffer>(FunId, TenIdin_trans, TenIdout_trans);
}

void newmaxpool(IdT FunId, IdT TenIdin, IdT TenIdout, uint32_t batch, uint32_t channel,uint32_t input_height, uint32_t input_width,uint32_t output_height, uint32_t output_width, uint32_t filter_height, uint32_t filter_width, uint32_t row_stride, uint32_t col_stride, uint32_t row_pad, uint32_t col_pad){
    shared_ptr<SecretTen > ten_in = GetTenById(TenIdin);                      
    shared_ptr<SecretTen > ten_out = GetTenById(TenIdout);
	IdT TenIdin_trans = GetBufferByIdM(FunId)->get_TenIdin_trans();
	shared_ptr<SecretTen> ten_in_trans = GetTenById(TenIdin_trans);
	IdT TenIdout_trans = GetBufferByIdM(FunId)->get_TenIdout_trans();
    shared_ptr<SecretTen> ten_out_trans = GetTenById(TenIdout_trans);  
    // printf("newmaxpool TenIdin_trans %ld, TenIdout_trans %ld\n", TenIdin_trans, TenIdout_trans);
	GetBufferByIdM(FunId)->forward(ten_in, ten_out,ten_in_trans, ten_out_trans, batch, channel,input_height,input_width,output_height,output_width,filter_height,filter_width,row_stride,col_stride);
}


unordered_map<IdT, shared_ptr<BatchnormBuffer>> BatchnormHolder;
shared_ptr<BatchnormBuffer> GetBufferByIdB(IdT FunId) {
    return BatchnormHolder[FunId];
}
    
void SecretInitBatchnorm(
        IdT FunId,
        IdT input, IdT output, IdT gamma, IdT beta,
        // IdT der_input, IdT der_output, IdT der_gamma, IdT der_beta,
        IdT run_mean, IdT run_var, IdT cur_mean, IdT cur_var,
        IdT mu,
        uint32_t batch_, uint32_t channel_, uint32_t height_, uint32_t width_,
        int affine_, int is_cumulative_, float momentum_, float epsilon_) {

    auto bn_buffer = make_shared<BatchnormBuffer>(FunId);
    BatchnormHolder[FunId] = bn_buffer;

    bn_buffer->init(
            input, output, gamma, beta,
            // der_input, der_output, der_gamma, der_beta,
            run_mean, run_var, cur_mean, cur_var,
            mu,
            batch_, channel_, height_, width_,
            affine_, is_cumulative_, momentum_, epsilon_);
}

void SecretBatchnormForward(IdT FunId, int Training) {
    GetBufferByIdB(FunId)->forward(Training);
}

unordered_map<IdT, shared_ptr<SGXLinearBuffer>> SGXLinearHolder;
shared_ptr<SGXLinearBuffer> GetSGXLinearBufferByIdB(IdT FunId) {
    return SGXLinearHolder[FunId];
}
void SecretInitSGXLinear(
        IdT FunId,
        IdT input, IdT output, IdT weight, IdT bias,
        // IdT der_input, IdT der_output, IdT der_weight, IdT der_bias,
        uint32_t batch_, uint32_t input_size_, uint32_t output_size_) {

    auto sgx_linear_buffer = make_shared<SGXLinearBuffer>(FunId);
    SGXLinearHolder[FunId] = sgx_linear_buffer;

    sgx_linear_buffer->init(
            input, output, weight, bias,
            // der_input, der_output, der_weight, der_bias,
            batch_, input_size_, output_size_);
}

void SecretSGXLinearForward(IdT FunId) {
    GetSGXLinearBufferByIdB(FunId)->forward();
}

unordered_map<IdT, shared_ptr<SGXConvBuffer>> SGXConvHolder;
shared_ptr<SGXConvBuffer> GetSGXConvBufferByIdB(IdT FunId) {
    return SGXConvHolder[FunId];
}
void SecretInitSGXConv(
        IdT FunId,
        IdT input, IdT output, IdT weight, IdT bias, 
        // IdT der_input, IdT der_output, IdT der_weight, IdT der_bias,
        uint32_t batch_, uint32_t input_h, uint32_t input_w, uint32_t input_c, 
        uint32_t output_h, uint32_t output_w, uint32_t output_c,
        uint32_t kernel, uint32_t padding, uint32_t stride) {

    auto sgx_conv_buffer = make_shared<SGXConvBuffer>(FunId);
    SGXConvHolder[FunId] = sgx_conv_buffer;
    sgx_conv_buffer->init(
            input, output, weight, bias, 
            // der_input, der_output, der_weight, der_bias,
            batch_, input_h, input_w, input_c, 
            output_h, output_w, output_c,
            kernel, padding, stride);
}

void SecretSGXConvForward(IdT FunId) {
    GetSGXConvBufferByIdB(FunId)->forward();
}

void SecretStochasticQuantize(IdT src_id, IdT dst_id, uint64_t q_tag) {
    quantize_stochastic(GetTenById(src_id), GetTenById(dst_id), q_tag);
}

} // End of extern C
