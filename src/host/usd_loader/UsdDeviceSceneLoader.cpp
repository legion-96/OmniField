#include "UsdDeviceSceneLoader.h"
#include <fstream>
using namespace omnifield; namespace sdfcad {
bool UsdDeviceSceneLoader::loadFromFile(const std::string& path, DeviceSceneHost& outScene, LoaderStats* stats){
  std::ifstream f(path); if(!f.good()){ std::cerr<<"[USD Loader] File not found: "<<path<<"\n"; return false; }
  outScene.clear();
  outScene.nodeTypes.push_back(NodeType::PRIMITIVE_SPHERE);
  outScene.nodeFlags.push_back(NODE_EXACT_SDF | NODE_ANALYTICAL_GRAD | NODE_BOUNDED);
  outScene.firstChild.push_back(0); outScene.childCount.push_back(0); outScene.objId.push_back(1);
  outScene.paramOffset.push_back(0); float radius=1.0f; const uint8_t* rp=reinterpret_cast<const uint8_t*>(&radius);
  outScene.paramBlob.insert(outScene.paramBlob.end(), rp, rp+sizeof(float));
  outScene.bounds.push_back({{-1.5f,-1.5f,-1.5f},{1.5f,1.5f,1.5f}}); outScene.lipschitzL.push_back(1.0f);
  for(int i=0;i<12;++i) outScene.xforms.push_back((i%5==0)?1.f:0.f);
  outScene.maxScale.push_back(1.0f); outScene.xformIsIdentity.push_back(1); outScene.nodeEpsilon.push_back(1e-4f);
  if(stats) stats->nodes=1; return true;
}}
