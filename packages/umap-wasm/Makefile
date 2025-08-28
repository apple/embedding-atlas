SOURCES = src/umap.cpp src/knn.cpp src/memory.cpp
HEADERS = src/common.hpp src/knn.hpp src/umap.hpp src/knncolle_nndescent.hpp src/distances.hpp

CXXFLAGS = -std=c++20 -O3 -Wall -Wno-deprecated-declarations

runtime.js: $(SOURCES) $(HEADERS)
	emcc $(CXXFLAGS) $(SOURCES) \
		third_party/nndescent/src/distances.cpp \
		third_party/nndescent/src/dtypes.cpp \
		third_party/nndescent/src/nnd.cpp \
		third_party/nndescent/src/rp_trees.cpp \
		third_party/nndescent/src/utils.cpp \
		-I third_party/umappp/include \
		-I third_party/knncolle/include \
		-I third_party/knncolle_hnsw/include \
		-I third_party/CppIrlba/include \
		-I third_party/CppKmeans/include \
		-I third_party/aarand/include \
		-I third_party/subpar/include \
		-I third_party/nndescent/src \
		-I third_party/Eigen \
		-I third_party/hnswlib \
		-DUMAPPP_NO_PARALLEL_OPTIMIZATION \
		-sENVIRONMENT=web \
		-sMODULARIZE -sSINGLE_FILE -sALLOW_MEMORY_GROWTH \
		-sEXPORTED_RUNTIME_METHODS='["HEAP32", "HEAPF32", "HEAPU8"]' \
		-sWASM_BIGINT=1 \
		-msimd128 \
		-o runtime.mjs
	mv runtime.mjs runtime.js

clean:
	rm -rf runtime.js
