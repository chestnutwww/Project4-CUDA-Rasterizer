/**
 * @file      rasterize.cu
 * @brief     CUDA-accelerated rasterization pipeline.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya, Shuai Shao (Shrek)
 * @date      2012-2016
 * @copyright University of Pennsylvania & STUDENT
 */

#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/random.h>
#include <util/checkCUDAError.h>
#include <util/tiny_gltf_loader.h>
#include "rasterizeTools.h"
#include "rasterize.h"
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#define MAXCOORD 999999.0f
#define MAXDEPTH 999999
#define POINTMODE 0
#define LINEMODE 0
#define TRIANGLEMODE 1
#define BACKFACECULLING 0

namespace {

	typedef unsigned short VertexIndex;
	typedef glm::vec3 VertexAttributePosition;
	typedef glm::vec3 VertexAttributeNormal;
	typedef glm::vec2 VertexAttributeTexcoord;
	typedef unsigned char TextureData;

	typedef unsigned char BufferByte;

	enum PrimitiveType{
		Point = 1,
		Line = 2,
		Triangle = 3
	};

	struct VertexOut {
		glm::vec4 pos;

		// TODO: add new attributes to your VertexOut
		// The attributes listed below might be useful, 
		// but always feel free to modify on your own

		 glm::vec3 eyePos;	// eye space position used for shading
		 glm::vec3 eyeNor;	// eye space normal used for shading, cuz normal will go wrong after perspective transformation
		 glm::vec3 color;
		 glm::vec2 screenPos;
		 glm::vec2 texcoord0;
		 TextureData* dev_diffuseTex = NULL;
		 int diffuseTexWidth;
		 int diffuseTexHeight;
		// ...
	};

	struct Primitive {
#if TRIANGLEMODE || LINEMODE
		PrimitiveType primitiveType = Triangle;	// C++ 11 init
		VertexOut v[3];
#endif
#if POINTMODE
		PrimitiveType primitiveType = Point;	// C++ 11 init
		VertexOut v[1];
#endif	
	};

	struct Fragment {
		glm::vec3 color;
		glm::vec3 eyeNor;
		glm::vec3 eyePos;
		// TODO: add new attributes to your Fragment
		// The attributes listed below might be useful, 
		// but always feel free to modify on your own

		// glm::vec3 eyePos;	// eye space position used for shading
		// glm::vec3 eyeNor;
		VertexAttributeTexcoord texcoord0;
		TextureData* dev_diffuseTex;
		int diffuseTexWidth;
		int diffuseTexHeight;
		// ...
	};

	struct PrimitiveDevBufPointers {
		int primitiveMode;	//from tinygltfloader macro
		PrimitiveType primitiveType;
		int numPrimitives;
		int numIndices;
		int numVertices;

		// Vertex In, const after loaded
		VertexIndex* dev_indices;
		VertexAttributePosition* dev_position;
		VertexAttributeNormal* dev_normal;
		VertexAttributeTexcoord* dev_texcoord0;

		// Materials, add more attributes when needed
		TextureData* dev_diffuseTex;
		int diffuseTexWidth;
		int diffuseTexHeight;
		// TextureData* dev_specularTex;
		// TextureData* dev_normalTex;
		// ...

		// Vertex Out, vertex used for rasterization, this is changing every frame
		VertexOut* dev_verticesOut;

		// TODO: add more attributes when needed
	};

}

/*
#if BACKFACECULLING
struct BackFace {
	__host__ __device__
		bool operator()(const Primitive &primitive) {
		return glm::dot(primitive.v[0].eyePos, primitive.v[0].eyeNor) > 0;
	}
};
#endif
*/

static std::map<std::string, std::vector<PrimitiveDevBufPointers>> mesh2PrimitivesMap;


static int width = 0;
static int height = 0;

static int totalNumPrimitives = 0;
static Primitive *dev_primitives = NULL;
static Fragment *dev_fragmentBuffer = NULL;
static glm::vec3 *dev_framebuffer = NULL;

static int * dev_depth = NULL;	// you might need this buffer when doing depth test
int * dev_fragMutex = NULL;

/**
 * Kernel that writes the image to the OpenGL PBO directly.
 */
__global__ 
void sendImageToPBO(uchar4 *pbo, int w, int h, glm::vec3 *image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
        glm::vec3 color;
        color.x = glm::clamp(image[index].x, 0.0f, 1.0f) * 255.0;
        color.y = glm::clamp(image[index].y, 0.0f, 1.0f) * 255.0;
        color.z = glm::clamp(image[index].z, 0.0f, 1.0f) * 255.0;
        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

__host__ __device__ static
glm::vec3 lambertShading(Fragment *fragmentBuffer, int index, glm::vec3 light_eyePos) {
	glm::vec3 I = glm::normalize(light_eyePos - fragmentBuffer[index].eyePos);
	float lambert = glm::max(glm::dot(I, fragmentBuffer[index].eyeNor), 0.0f);
	return lambert * fragmentBuffer[index].color;
}

__host__ __device__ static
glm::vec3 blinnPhongShading(Fragment *fragmentBuffer, int index, glm::vec3 light_eyePos) {
	float u_shininess = 100.0f;	
	glm::vec3 I = glm::normalize(light_eyePos - fragmentBuffer[index].eyePos);
	glm::vec3 H = glm::normalize((I - glm::normalize(fragmentBuffer[index].eyePos)) / 2.0f);
	float lambert = glm::max(glm::dot(I, fragmentBuffer[index].eyeNor), 0.0f);
	float specular = glm::max(glm::pow(glm::dot(H, fragmentBuffer[index].eyeNor), u_shininess), 0.0f);

	glm::vec3 frag_color = lambert * fragmentBuffer[index].color;
	frag_color += specular * frag_color;

	return frag_color;
}

__host__ __device__ static
glm::vec3 toonShading(Fragment *fragmentBuffer, int index, glm::vec3 light_eyePos) {
	float steps = 3.0f; // used to compute discrete lambert and specular
	float toonEffect = 0.7f; // used to control how much effect of toon you want, range [0, 1]
	
	// ----------------------Lambert part----------------------
	// Compute original lambert and toon(discrete lambert)
	// Use lerp to decide the how much effect of toon you want
	glm::vec3 I = glm::normalize(light_eyePos - fragmentBuffer[index].eyePos);
	float lambert = glm::max(glm::dot(I, fragmentBuffer[index].eyeNor), 0.0f);
	float toon = floor(lambert * steps) / steps;
	lambert = lambert * (1 - toonEffect) + toon * toonEffect;

	// ----------------------Specular part----------------------
	// Compute original specular and toon(discrete lambert)
	// Use lerp to decide the how much effect of toon you want
	float u_shininess = 32.0f;
	float atten = 1 / glm::length(light_eyePos - fragmentBuffer[index].eyePos);
	glm::vec3 H = glm::normalize((I - glm::normalize(fragmentBuffer[index].eyePos)) / 2.0f);
	float specular = glm::max(glm::pow(glm::dot(H, fragmentBuffer[index].eyeNor), u_shininess), 0.0f);
	float toonSpec = floor(specular * atten * 2) / 2;
	specular = specular * (1 - toonEffect) + toon * toonEffect;

	// Compute final color
	glm::vec3 frag_color = lambert * fragmentBuffer[index].color;
	frag_color += specular * frag_color;
	return frag_color;
}

/** 
* Writes fragment colors to the framebuffer
*/
__global__
void render(int w, int h, Fragment *fragmentBuffer, glm::vec3 *framebuffer) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
#if POINTMODE || LINEMODE       
		framebuffer[index] = fragmentBuffer[index].color;
#endif
		// TODO: add your fragment shader code here
#if TRIANGLEMODE		
		glm::vec3 frag_color(0.0f, 0.0f, 0.0f);
		if (fragmentBuffer[index].dev_diffuseTex != NULL)
		{
			float dirx[4] = { 0.5f, 0.5f, -0.5f, 0.5f };
			float diry[4] = { 0.5f, -0.5f, 0.5f, -0.5f };
			for (int i = 0; i < 4; ++i) {	
				int tex_x = (fragmentBuffer[index].diffuseTexWidth - 1) * fragmentBuffer[index].texcoord0.x + dirx[i];
				int tex_y = (fragmentBuffer[index].diffuseTexHeight - 1) * fragmentBuffer[index].texcoord0.y + diry[i];
				int tex_id = tex_y * fragmentBuffer[index].diffuseTexWidth + tex_x;
				glm::vec3 tex_color(
					fragmentBuffer[index].dev_diffuseTex[3 * tex_id],
					fragmentBuffer[index].dev_diffuseTex[3 * tex_id + 1],
					fragmentBuffer[index].dev_diffuseTex[3 * tex_id + 2]);
				tex_color /= 255.0f;
				frag_color += tex_color;
			}
			fragmentBuffer[index].color = frag_color / 4.0f;
		}

		glm::vec3 light_eyePos(50, 50, 100);

		// ---------------------------------------------------------------------
		// ------------------------- Shading Part ------------------------------
		// ------------ Use comment to decide which shading to uese ------------
		// ---------------------------------------------------------------------
		// 1. Lambert Shading
		//framebuffer[index] = lambertShading(fragmentBuffer, index, light_eyePos);
		
		// 2. Blinn-Phong Shading
		//framebuffer[index] = blinnPhongShading(fragmentBuffer, index, light_eyePos);

		// 3. ToonShading
		framebuffer[index] = toonShading(fragmentBuffer, index, light_eyePos);
		// ---------------------------------------------------------------------


#endif		
		
    }
}

/**
 * Called once at the beginning of the program to allocate memory.
 */
void rasterizeInit(int w, int h) {
    width = w;
    height = h;
	cudaFree(dev_fragmentBuffer);
	cudaMalloc(&dev_fragmentBuffer, width * height * sizeof(Fragment));
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
    cudaFree(dev_framebuffer);
    cudaMalloc(&dev_framebuffer,   width * height * sizeof(glm::vec3));
    cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));
    
	cudaFree(dev_depth);
	cudaMalloc(&dev_depth, width * height * sizeof(int));
	cudaMemset(dev_depth, MAXDEPTH, height * sizeof(int));

	cudaMalloc(&dev_fragMutex, width * height * sizeof(int));
	cudaMemset(dev_fragMutex, 0, height * sizeof(int));

	checkCUDAError("rasterizeInit");
}

__global__
void initDepth(int w, int h, int * depth)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < w && y < h)
	{
		int index = x + (y * w);
		depth[index] = INT_MAX;
	}
}


/**
* kern function with support for stride to sometimes replace cudaMemcpy
* One thread is responsible for copying one component
*/
__global__ 
void _deviceBufferCopy(int N, BufferByte* dev_dst, const BufferByte* dev_src, int n, int byteStride, int byteOffset, int componentTypeByteSize) {
	
	// Attribute (vec3 position)
	// component (3 * float)
	// byte (4 * byte)

	// id of component
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (i < N) {
		int count = i / n;
		int offset = i - count * n;	// which component of the attribute

		for (int j = 0; j < componentTypeByteSize; j++) {
			
			dev_dst[count * componentTypeByteSize * n 
				+ offset * componentTypeByteSize 
				+ j]

				= 

			dev_src[byteOffset 
				+ count * (byteStride == 0 ? componentTypeByteSize * n : byteStride) 
				+ offset * componentTypeByteSize 
				+ j];
		}
	}
	

}

__global__
void _nodeMatrixTransform(
	int numVertices,
	VertexAttributePosition* position,
	VertexAttributeNormal* normal,
	glm::mat4 MV, glm::mat3 MV_normal) {

	// vertex id
	int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (vid < numVertices) {
		position[vid] = glm::vec3(MV * glm::vec4(position[vid], 1.0f));
		normal[vid] = glm::normalize(MV_normal * normal[vid]);
	}
}

glm::mat4 getMatrixFromNodeMatrixVector(const tinygltf::Node & n) {
	
	glm::mat4 curMatrix(1.0);

	const std::vector<double> &m = n.matrix;
	if (m.size() > 0) {
		// matrix, copy it

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				curMatrix[i][j] = (float)m.at(4 * i + j);
			}
		}
	} else {
		// no matrix, use rotation, scale, translation

		if (n.translation.size() > 0) {
			curMatrix[3][0] = n.translation[0];
			curMatrix[3][1] = n.translation[1];
			curMatrix[3][2] = n.translation[2];
		}

		if (n.rotation.size() > 0) {
			glm::mat4 R;
			glm::quat q;
			q[0] = n.rotation[0];
			q[1] = n.rotation[1];
			q[2] = n.rotation[2];

			R = glm::mat4_cast(q);
			curMatrix = curMatrix * R;
		}

		if (n.scale.size() > 0) {
			curMatrix = curMatrix * glm::scale(glm::vec3(n.scale[0], n.scale[1], n.scale[2]));
		}
	}

	return curMatrix;
}

void traverseNode (
	std::map<std::string, glm::mat4> & n2m,
	const tinygltf::Scene & scene,
	const std::string & nodeString,
	const glm::mat4 & parentMatrix
	) 
{
	const tinygltf::Node & n = scene.nodes.at(nodeString);
	glm::mat4 M = parentMatrix * getMatrixFromNodeMatrixVector(n);
	n2m.insert(std::pair<std::string, glm::mat4>(nodeString, M));

	auto it = n.children.begin();
	auto itEnd = n.children.end();

	for (; it != itEnd; ++it) {
		traverseNode(n2m, scene, *it, M);
	}
}

void rasterizeSetBuffers(const tinygltf::Scene & scene) {

	totalNumPrimitives = 0;

	std::map<std::string, BufferByte*> bufferViewDevPointers;

	// 1. copy all `bufferViews` to device memory
	{
		std::map<std::string, tinygltf::BufferView>::const_iterator it(
			scene.bufferViews.begin());
		std::map<std::string, tinygltf::BufferView>::const_iterator itEnd(
			scene.bufferViews.end());

		for (; it != itEnd; it++) {
			const std::string key = it->first;
			const tinygltf::BufferView &bufferView = it->second;
			if (bufferView.target == 0) {
				continue; // Unsupported bufferView.
			}

			const tinygltf::Buffer &buffer = scene.buffers.at(bufferView.buffer);

			BufferByte* dev_bufferView;
			cudaMalloc(&dev_bufferView, bufferView.byteLength);
			cudaMemcpy(dev_bufferView, &buffer.data.front() + bufferView.byteOffset, bufferView.byteLength, cudaMemcpyHostToDevice);

			checkCUDAError("Set BufferView Device Mem");

			bufferViewDevPointers.insert(std::make_pair(key, dev_bufferView));

		}
	}



	// 2. for each mesh: 
	//		for each primitive: 
	//			build device buffer of indices, materail, and each attributes
	//			and store these pointers in a map
	{

		std::map<std::string, glm::mat4> nodeString2Matrix;
		auto rootNodeNamesList = scene.scenes.at(scene.defaultScene);

		{
			auto it = rootNodeNamesList.begin();
			auto itEnd = rootNodeNamesList.end();
			for (; it != itEnd; ++it) {
				traverseNode(nodeString2Matrix, scene, *it, glm::mat4(1.0f));
			}
		}


		// parse through node to access mesh

		auto itNode = nodeString2Matrix.begin();
		auto itEndNode = nodeString2Matrix.end();
		for (; itNode != itEndNode; ++itNode) {

			const tinygltf::Node & N = scene.nodes.at(itNode->first);
			const glm::mat4 & matrix = itNode->second;
			const glm::mat3 & matrixNormal = glm::transpose(glm::inverse(glm::mat3(matrix)));

			auto itMeshName = N.meshes.begin();
			auto itEndMeshName = N.meshes.end();

			for (; itMeshName != itEndMeshName; ++itMeshName) {

				const tinygltf::Mesh & mesh = scene.meshes.at(*itMeshName);

				auto res = mesh2PrimitivesMap.insert(std::pair<std::string, std::vector<PrimitiveDevBufPointers>>(mesh.name, std::vector<PrimitiveDevBufPointers>()));
				std::vector<PrimitiveDevBufPointers> & primitiveVector = (res.first)->second;

				// for each primitive
				for (size_t i = 0; i < mesh.primitives.size(); i++) {
					const tinygltf::Primitive &primitive = mesh.primitives[i];

					if (primitive.indices.empty())
						return;

					// TODO: add new attributes for your PrimitiveDevBufPointers when you add new attributes
					VertexIndex* dev_indices = NULL;
					VertexAttributePosition* dev_position = NULL;
					VertexAttributeNormal* dev_normal = NULL;
					VertexAttributeTexcoord* dev_texcoord0 = NULL;

					// ----------Indices-------------

					const tinygltf::Accessor &indexAccessor = scene.accessors.at(primitive.indices);
					const tinygltf::BufferView &bufferView = scene.bufferViews.at(indexAccessor.bufferView);
					BufferByte* dev_bufferView = bufferViewDevPointers.at(indexAccessor.bufferView);

					// assume type is SCALAR for indices
					int n = 1;
					int numIndices = indexAccessor.count;
					int componentTypeByteSize = sizeof(VertexIndex);
					int byteLength = numIndices * n * componentTypeByteSize;

					dim3 numThreadsPerBlock(128);
					dim3 numBlocks((numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
					cudaMalloc(&dev_indices, byteLength);
					_deviceBufferCopy << <numBlocks, numThreadsPerBlock >> > (
						numIndices,
						(BufferByte*)dev_indices,
						dev_bufferView,
						n,
						indexAccessor.byteStride,
						indexAccessor.byteOffset,
						componentTypeByteSize);

					checkCUDAError("Set Index Buffer");

					// ---------Primitive Info-------

					// Warning: LINE_STRIP is not supported in tinygltfloader
					int numPrimitives;
					PrimitiveType primitiveType;
#if TRIANGLEMODE || LINEMODE
					switch (primitive.mode) {
					case TINYGLTF_MODE_TRIANGLES:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices / 3;
						break;
					case TINYGLTF_MODE_TRIANGLE_STRIP:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices - 2;
						break;
					case TINYGLTF_MODE_TRIANGLE_FAN:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices - 2;
						break;
					case TINYGLTF_MODE_LINE:
						primitiveType = PrimitiveType::Line;
						numPrimitives = numIndices / 2;
						break;
					case TINYGLTF_MODE_LINE_LOOP:
						primitiveType = PrimitiveType::Line;
						numPrimitives = numIndices + 1;
						break;
					case TINYGLTF_MODE_POINTS:
						primitiveType = PrimitiveType::Point;
						numPrimitives = numIndices;
						break;
					default:
						// output error
						break;
					};
#endif
#if POINTMODE
					primitiveType = PrimitiveType::Point;
					numPrimitives = numIndices;
#endif
					// ----------Attributes-------------

					auto it(primitive.attributes.begin());
					auto itEnd(primitive.attributes.end());

					int numVertices = 0;
					// for each attribute
					for (; it != itEnd; it++) {
						const tinygltf::Accessor &accessor = scene.accessors.at(it->second);
						const tinygltf::BufferView &bufferView = scene.bufferViews.at(accessor.bufferView);

						int n = 1;
						if (accessor.type == TINYGLTF_TYPE_SCALAR) {
							n = 1;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC2) {
							n = 2;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC3) {
							n = 3;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC4) {
							n = 4;
						}

						BufferByte * dev_bufferView = bufferViewDevPointers.at(accessor.bufferView);
						BufferByte ** dev_attribute = NULL;

						numVertices = accessor.count;
						int componentTypeByteSize;

						// Note: since the type of our attribute array (dev_position) is static (float32)
						// We assume the glTF model attribute type are 5126(FLOAT) here

						if (it->first.compare("POSITION") == 0) {
							componentTypeByteSize = sizeof(VertexAttributePosition) / n;
							dev_attribute = (BufferByte**)&dev_position;
						}
						else if (it->first.compare("NORMAL") == 0) {
							componentTypeByteSize = sizeof(VertexAttributeNormal) / n;
							dev_attribute = (BufferByte**)&dev_normal;
						}
						else if (it->first.compare("TEXCOORD_0") == 0) {
							componentTypeByteSize = sizeof(VertexAttributeTexcoord) / n;
							dev_attribute = (BufferByte**)&dev_texcoord0;
						}

						std::cout << accessor.bufferView << "  -  " << it->second << "  -  " << it->first << '\n';

						dim3 numThreadsPerBlock(128);
						dim3 numBlocks((n * numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
						int byteLength = numVertices * n * componentTypeByteSize;
						cudaMalloc(dev_attribute, byteLength);

						_deviceBufferCopy << <numBlocks, numThreadsPerBlock >> > (
							n * numVertices,
							*dev_attribute,
							dev_bufferView,
							n,
							accessor.byteStride,
							accessor.byteOffset,
							componentTypeByteSize);

						std::string msg = "Set Attribute Buffer: " + it->first;
						checkCUDAError(msg.c_str());
					}

					// malloc for VertexOut
					VertexOut* dev_vertexOut;
					cudaMalloc(&dev_vertexOut, numVertices * sizeof(VertexOut));
					checkCUDAError("Malloc VertexOut Buffer");

					// ----------Materials-------------

					// You can only worry about this part once you started to 
					// implement textures for your rasterizer
					TextureData* dev_diffuseTex = NULL;
					int diffuseTexWidth = 0;
					int diffuseTexHeight = 0;
					if (!primitive.material.empty()) {
						const tinygltf::Material &mat = scene.materials.at(primitive.material);
						printf("material.name = %s\n", mat.name.c_str());

						if (mat.values.find("diffuse") != mat.values.end()) {
							std::string diffuseTexName = mat.values.at("diffuse").string_value;
							if (scene.textures.find(diffuseTexName) != scene.textures.end()) {
								const tinygltf::Texture &tex = scene.textures.at(diffuseTexName);
								if (scene.images.find(tex.source) != scene.images.end()) {
									const tinygltf::Image &image = scene.images.at(tex.source);

									size_t s = image.image.size() * sizeof(TextureData);
									cudaMalloc(&dev_diffuseTex, s);
									cudaMemcpy(dev_diffuseTex, &image.image.at(0), s, cudaMemcpyHostToDevice);
									
									diffuseTexWidth = image.width;
									diffuseTexHeight = image.height;

									checkCUDAError("Set Texture Image data");
								}
							}
						}

						// TODO: write your code for other materails
						// You may have to take a look at tinygltfloader
						// You can also use the above code loading diffuse material as a start point 
					}


					// ---------Node hierarchy transform--------
					cudaDeviceSynchronize();
					
					dim3 numBlocksNodeTransform((numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
					_nodeMatrixTransform << <numBlocksNodeTransform, numThreadsPerBlock >> > (
						numVertices,
						dev_position,
						dev_normal,
						matrix,
						matrixNormal);

					checkCUDAError("Node hierarchy transformation");

					// at the end of the for loop of primitive
					// push dev pointers to map
					primitiveVector.push_back(PrimitiveDevBufPointers{
						primitive.mode,
						primitiveType,
						numPrimitives,
						numIndices,
						numVertices,

						dev_indices,
						dev_position,
						dev_normal,
						dev_texcoord0,

						dev_diffuseTex,
						diffuseTexWidth,
						diffuseTexHeight,

						dev_vertexOut	//VertexOut
					});

					totalNumPrimitives += numPrimitives;

				} // for each primitive

			} // for each mesh

		} // for each node

	}
	

	// 3. Malloc for dev_primitives
	{
		cudaMalloc(&dev_primitives, totalNumPrimitives * sizeof(Primitive));
	}
	

	// Finally, cudaFree raw dev_bufferViews
	{

		std::map<std::string, BufferByte*>::const_iterator it(bufferViewDevPointers.begin());
		std::map<std::string, BufferByte*>::const_iterator itEnd(bufferViewDevPointers.end());
			
			//bufferViewDevPointers

		for (; it != itEnd; it++) {
			cudaFree(it->second);
		}

		checkCUDAError("Free BufferView Device Mem");
	}


}



__global__ 
void _vertexTransformAndAssembly(
	int numVertices, 
	PrimitiveDevBufPointers primitive, 
	glm::mat4 MVP, glm::mat4 MV, glm::mat3 MV_normal, 
	int width, int height) {

	// vertex id
	int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (vid < numVertices) {

		// TODO: Apply vertex transformation here
		// Multiply the MVP matrix for each vertex position, this will transform everything into clipping space
		// Then divide the pos by its w element to transform into NDC space
		// Finally transform x and y to viewport space
		
		// 1. world -> unhomogeneous screen (get world pos from VertexIn)
		glm::vec4 clip_position4 = MVP * glm::vec4(primitive.dev_position[vid], 1);
		// 2. homogeneous screen -> screen
		clip_position4 /= clip_position4[3];
		// 3. screen -> ndc (update texcoord in VertexOut)
		glm::vec2 ndc_position = glm::vec2(((clip_position4[0] + 1) / 2) * width, ((1 - clip_position4[1]) / 2) * height);
		// 4. update camera space pos/norm in VertexOut
		glm::vec3 camera_position = multiplyMV(MV, glm::vec4(primitive.dev_position[vid], 1));
		glm::vec3 camera_normal = glm::normalize(MV_normal * primitive.dev_normal[vid]);

		// TODO: Apply vertex assembly here
		// Assemble all attribute arraies into the primitive array
		primitive.dev_verticesOut[vid].pos = clip_position4;
		primitive.dev_verticesOut[vid].screenPos = ndc_position;
		primitive.dev_verticesOut[vid].eyePos = camera_position;
		primitive.dev_verticesOut[vid].eyeNor = camera_normal;
		if (primitive.dev_texcoord0 != NULL)
		{
			primitive.dev_verticesOut[vid].texcoord0 = primitive.dev_texcoord0[vid];
			primitive.dev_verticesOut[vid].dev_diffuseTex = primitive.dev_diffuseTex;
			primitive.dev_verticesOut[vid].diffuseTexWidth = primitive.diffuseTexWidth;
			primitive.dev_verticesOut[vid].diffuseTexHeight = primitive.diffuseTexHeight;
		}
			

	}
}



static int curPrimitiveBeginId = 0;

__global__ 
void _primitiveAssembly(int numIndices, int curPrimitiveBeginId, Primitive* dev_primitives, PrimitiveDevBufPointers primitive) {

	// index id
	int iid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (iid < numIndices) {

		// TODO: uncomment the following code for a start
		// This is primitive assembly for triangles

		int pid;	// id for cur primitives vector
		//if (primitive.primitiveMode == TINYGLTF_MODE_TRIANGLES) {
			pid = iid / (int)primitive.primitiveType;
			dev_primitives[pid + curPrimitiveBeginId].v[iid % (int)primitive.primitiveType]
				= primitive.dev_verticesOut[primitive.dev_indices[iid]];
		//}
		// TODO: other primitive types (point, line)
	}
	
}


__host__ __device__ static
bool epsilonCheck(double a, double b) {
	if (fabs(fabs(a) - fabs(b)) < 0.01f) {
		return true;
	}
	else {
		return false;
	}
}


__global__
void _rasterization(
	int numPrimitives,
	Primitive* dev_primitives,
	Fragment *dev_fragment,
	int *dev_depth,
	int *dev_fragMutex,
	int width, int height) {

	//primitives id
	int pid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (pid < numPrimitives) {

#if POINTMODE
		VertexOut point = dev_primitives[pid].v[0];	
		glm::vec3 point_color = glm::vec3(1.0f, 1.0f, 1.0f);
		int cur_depth = 200 * point.eyePos.z;
		int fid = int(point.screenPos.y) * width + int(point.screenPos.x);
		if (cur_depth < dev_depth[fid]) {
			dev_depth[fid] = cur_depth;
			dev_fragment[fid].color = point_color;
		}
#endif

#if LINEMODE
		VertexOut line_points[3];
		for (int i = 0; i < 3; ++i)
		{
			line_points[i] = dev_primitives[pid].v[i];
			line_points[i].color = glm::vec3(1.0f, 1.0f, 1.0f);
		}

		// For each edge in a triangle, use lerp to set color
		float t = 0.5f;
		for (int i = 0; i < 3; ++i)
		{	
			// 1. Initialize 2 points for current loop
			int p0 = i;
			int p1 = (i + 1) % 3;
			glm::vec2 cur_pos = line_points[p0].screenPos;
			float cur_distance = 0.0f;
			float distance = fabs(glm::length(line_points[p1].screenPos - line_points[p0].screenPos));
			
			// 2. move from p0 to p1
			while (distance > cur_distance) {
				// move
				glm::vec2 direction = glm::normalize(line_points[p1].screenPos - line_points[p0].screenPos);
				cur_pos += t * direction;
				cur_distance = fabs(glm::length(cur_pos - line_points[p0].screenPos));

				// use lerp to compute fragment id and color
				float lerp_coef = cur_distance / distance;
				glm::vec2 cur_screen = (1 - lerp_coef) * line_points[p0].screenPos + lerp_coef * line_points[p1].screenPos;
				glm::vec3 cur_color = (1 - lerp_coef) * line_points[p0].color + lerp_coef * line_points[p1].color;
				int fid = int(cur_screen.y) * width + int(cur_screen.x);
				//update
				dev_fragment[fid].color = cur_color;
			}
		}		
#endif

#if TRIANGLEMODE
		// 1. Get bounding box of the triangle in 2D screen space (the z axis-tri[0] has no meaning)
		//VertexOut tri_points[3];
		glm::vec3 tri_eye[3];
		glm::vec3 tri_screen[3];
		glm::vec3 tri_color[3];
		glm::vec3 tri_norm[3];
		glm::vec2 tri_texcoord0[3];
		for (int i = 0; i < 3; ++i) {
			tri_screen[i] = glm::vec3(dev_primitives[pid].v[i].screenPos, 0.0f);
			tri_eye[i] = dev_primitives[pid].v[i].eyePos;
			tri_norm[i] = dev_primitives[pid].v[i].eyeNor;
			//tri_color[i] = dev_primitives[pid].v[i].color;
			tri_color[i] = glm::vec3(0.5f, 0.5f, 0.5f);
			tri_texcoord0[i] = dev_primitives[pid].v[i].texcoord0;
		}

		AABB boundingbox = getAABBForTriangle(tri_screen);

#if BACKFACECULLING
		if (calculateSignedArea(tri_eye) > 0)
			return;
#endif

		// 2. loop over pixels within the bouding box
		for (int y = boundingbox.min.y; y <= boundingbox.max.y; ++y) {
			for (int x = boundingbox.min.x; x <= boundingbox.max.x; ++x) {

				int fid = y * width + x;
				glm::vec3 coef = calculateBarycentricCoordinate(tri_screen, glm::vec2(x, y));


				// check if the pixel lies in the triangle
				if (isBarycentricCoordInBounds(coef)) {
					glm::vec3 cur_color = tri_color[0] * coef[0] + tri_color[1] * coef[1] + tri_color[2] * coef[2];
					int cur_depth = 200 * getZAtCoordinate(coef, tri_eye);

					// depth test
					bool isSet = true;
					do {
						isSet = (atomicCAS(&dev_fragMutex[fid], 0, 1) == 0);
						if (isSet) {
							if (cur_depth < dev_depth[fid]) {
								dev_depth[fid] = cur_depth;
								dev_fragment[fid].color = cur_color;
								dev_fragment[fid].eyeNor = glm::normalize(tri_norm[0] * coef[0] + tri_norm[1] * coef[1] + tri_norm[2] * coef[2]);
								dev_fragment[fid].eyePos = tri_eye[0] * coef[0] + tri_eye[1] * coef[1] + tri_eye[2] * coef[2];
								dev_fragment[fid].dev_diffuseTex = dev_primitives[pid].v[0].dev_diffuseTex;
								dev_fragment[fid].texcoord0 = 
									tri_texcoord0[0] * coef[0] / tri_eye[0].z
									+ tri_texcoord0[1] * coef[1] / tri_eye[1].z
									+ tri_texcoord0[2] * coef[2] / tri_eye[2].z;
								dev_fragment[fid].texcoord0 *= getRealZAtCoordinate(coef, tri_eye);
								dev_fragment[fid].diffuseTexWidth = dev_primitives[pid].v[0].diffuseTexWidth;
								dev_fragment[fid].diffuseTexHeight = dev_primitives[pid].v[0].diffuseTexHeight;
							}
						}
						if (isSet) {
							dev_fragMutex[fid] = 0;
						}
					} while (!isSet);					
				}
			}
		}
#endif
	}
}


/**
 * Perform rasterization.
 */

static int callingcount = 0;

void rasterize(uchar4 *pbo, const glm::mat4 & MVP, const glm::mat4 & MV, const glm::mat3 MV_normal) {
	callingcount++;
    int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((width  - 1) / blockSize2d.x + 1,
		(height - 1) / blockSize2d.y + 1);

	// Execute your rasterization pipeline here
	// (See README for rasterization pipeline outline.)

	//-----cuda event for testing runtime-----
	/*
	//  1.create
	float time_vertexShader = 0.0f, t_vertexShaerPerIter;
	float time_primitiveAssembly = 0.0f, t_primitiveAssemblyPerIter;
	float time_rasterization = 0.0f, t_rasterizationPerIter;
	float time_render = 0.0f, t_render;
	cudaEvent_t start_vertexShader, stop_vertexShader;
	cudaEvent_t start_primitiveAssembly, stop_primitiveAssembly;
	cudaEvent_t start_rasterization, stop_rasterization;
	cudaEvent_t start_render, stop_render;
	cudaEventCreate(&start_vertexShader);
	cudaEventCreate(&stop_vertexShader);
	cudaEventCreate(&start_primitiveAssembly);
	cudaEventCreate(&stop_primitiveAssembly);
	cudaEventCreate(&start_rasterization);
	cudaEventCreate(&stop_rasterization);
	cudaEventCreate(&start_render);
	cudaEventCreate(&stop_render);
	*/
	//----------------------------------------

	// Vertex Process & primitive assembly
	{
		curPrimitiveBeginId = 0;
		dim3 numThreadsPerBlock(128);

		auto it = mesh2PrimitivesMap.begin();
		auto itEnd = mesh2PrimitivesMap.end();

		for (; it != itEnd; ++it) {
			auto p = (it->second).begin();	// each primitive
			auto pEnd = (it->second).end();
			for (; p != pEnd; ++p) {
				dim3 numBlocksForVertices((p->numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
				dim3 numBlocksForIndices((p->numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);

				//-----cuda event for testing runtime-----
				//cudaEventRecord(start_vertexShader);
				//----------------------------------------
				
				_vertexTransformAndAssembly << < numBlocksForVertices, numThreadsPerBlock >> >(p->numVertices, *p, MVP, MV, MV_normal, width, height);
				
				//-----cuda event for testing runtime-----
				/*
				cudaEventRecord(stop_vertexShader);
				cudaEventSynchronize(stop_vertexShader);
				cudaEventElapsedTime(&t_vertexShaerPerIter, start_vertexShader, stop_vertexShader);
				time_vertexShader += t_vertexShaerPerIter;
				*/
				//----------------------------------------

				checkCUDAError("Vertex Processing");
				cudaDeviceSynchronize();
				
				//-----cuda event for testing runtime-----
				//cudaEventRecord(start_primitiveAssembly);
				//----------------------------------------
				
				_primitiveAssembly << < numBlocksForIndices, numThreadsPerBlock >> >
					(p->numIndices, 
					curPrimitiveBeginId, 
					dev_primitives, 
					*p);
				
				//-----cuda event for testing runtime-----
				/*
				cudaEventRecord(stop_primitiveAssembly);
				cudaEventSynchronize(stop_primitiveAssembly);
				cudaEventElapsedTime(&t_primitiveAssemblyPerIter, start_primitiveAssembly, stop_primitiveAssembly);
				time_primitiveAssembly += t_primitiveAssemblyPerIter;
				*/
				//----------------------------------------
				
				checkCUDAError("Primitive Assembly");

				curPrimitiveBeginId += p->numPrimitives;
			}
		}

		checkCUDAError("Vertex Processing and Primitive Assembly");
	}
	
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));

	initDepth << <blockCount2d, blockSize2d >> >(width, height, dev_depth);

	// TODO: rasterize
/*
#if BACKFACECULLING
	Primitive *dev_primitives_end = thrust::remove_if(thrust::device, dev_primitives, dev_primitives + totalNumPrimitives, BackFace());
	totalNumPrimitives = dev_primitives_end - dev_primitives;
#endif
*/
	dim3 numThreadPerBlock(128);
	dim3 numBlocksForPrimitives((totalNumPrimitives + numThreadPerBlock.x - 1) / numThreadPerBlock.x);
	
	//-----cuda event for testing runtime-----
	//cudaEventRecord(start_rasterization);
	//----------------------------------------
	
	_rasterization << < numBlocksForPrimitives, numThreadPerBlock >> >
		(totalNumPrimitives,
			dev_primitives,
			dev_fragmentBuffer,
			dev_depth,
			dev_fragMutex,
			width, height);

	//-----cuda event for testing runtime-----
	/*
	cudaEventRecord(stop_rasterization);
	cudaEventSynchronize(stop_rasterization);
	cudaEventElapsedTime(&time_rasterization, start_rasterization, stop_rasterization);
	*/
	//----------------------------------------

	checkCUDAError("Rasterization");

	//-----cuda event for testing runtime-----
	//cudaEventRecord(start_render);
	//----------------------------------------

    // Copy depthbuffer colors into framebuffer
	render << <blockCount2d, blockSize2d >> >(width, height, dev_fragmentBuffer, dev_framebuffer);

	//-----cuda event for testing runtime-----
	/*
	cudaEventRecord(stop_render);
	cudaEventSynchronize(stop_render);
	cudaEventElapsedTime(&time_render, start_render, stop_render);
	*/
	//----------------------------------------

	checkCUDAError("fragment shader");
    // Copy framebuffer into OpenGL buffer for OpenGL previewing
    sendImageToPBO<<<blockCount2d, blockSize2d>>>(pbo, width, height, dev_framebuffer);
    checkCUDAError("copy render result to pbo");

	//-----cuda event for testing runtime-----
	/*
	if (callingcount == 1)
	{
		std::cout << "run time of vertexShaer: " << time_vertexShader << std::endl;
		std::cout << "run time of primitiveAssembly: " << time_primitiveAssembly << std::endl;
		std::cout << "run time of rasterization: " << time_rasterization << std::endl;
		std::cout << "run time of render: " << time_render << std::endl;
	}
	*/
	//----------------------------------------
}

/**
 * Called once at the end of the program to free CUDA memory.
 */
void rasterizeFree() {

    // deconstruct primitives attribute/indices device buffer

	auto it(mesh2PrimitivesMap.begin());
	auto itEnd(mesh2PrimitivesMap.end());
	for (; it != itEnd; ++it) {
		for (auto p = it->second.begin(); p != it->second.end(); ++p) {
			cudaFree(p->dev_indices);
			cudaFree(p->dev_position);
			cudaFree(p->dev_normal);
			cudaFree(p->dev_texcoord0);
			cudaFree(p->dev_diffuseTex);

			cudaFree(p->dev_verticesOut);

			
			//TODO: release other attributes and materials
		}
	}

	////////////

    cudaFree(dev_primitives);
    dev_primitives = NULL;

	cudaFree(dev_fragmentBuffer);
	dev_fragmentBuffer = NULL;

    cudaFree(dev_framebuffer);
    dev_framebuffer = NULL;

	cudaFree(dev_depth);
	dev_depth = NULL;

	cudaFree(dev_fragMutex);
	dev_fragMutex = NULL;

    checkCUDAError("rasterize Free");
}
