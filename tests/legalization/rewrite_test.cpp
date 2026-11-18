#include "vbx_rewrite/engine.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
void check(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
void info(onnx::ValueInfoProto* value,const char* name,std::initializer_list<int64_t> shape,int type=onnx::TensorProto::FLOAT) {
  value->set_name(name);auto* t=value->mutable_type()->mutable_tensor_type();t->set_elem_type(type);
  for(auto d:shape)t->mutable_shape()->add_dim()->set_dim_value(d);
}
onnx::ModelProto fixture(int64_t axis=1,int opset=17,int keep=1) {
  onnx::ModelProto m;m.set_ir_version(9);auto* o=m.add_opset_import();o->set_domain("");o->set_version(opset);
  auto* g=m.mutable_graph();g->set_name("test");info(g->add_input(),"x",{1,3,8,8});
  info(g->add_output(),"y",{1,1,8,8});auto* n=g->add_node();n->set_op_type("ReduceMean");
  n->set_name("mean");n->set_doc_string("preserve me");n->add_input("x");n->add_output("y");
  auto* a=n->add_attribute();a->set_name("keepdims");a->set_type(onnx::AttributeProto::INT);a->set_i(keep);
  if(opset<18) {a=n->add_attribute();a->set_name("axes");a->set_type(onnx::AttributeProto::INTS);a->add_ints(axis);}
  else {n->add_input("axes");auto* t=g->add_initializer();t->set_name("axes");t->set_data_type(onnx::TensorProto::INT64);t->add_dims(1);t->add_int64_data(axis);}
  return m;
}
void sameWhenSkipped(onnx::ModelProto m) {
  std::string before=m.SerializeAsString();auto stats=vbx::rewriteModel(m);
  check(stats.rewrites==0,"unexpected rewrite");check(before==m.SerializeAsString(),"failed match changed model");
}
}
int main() {
  try {
    for(auto axis:{int64_t(1),int64_t(-3)})for(auto opset:{11,17,18,21}) {
      auto m=fixture(axis,opset);auto s=vbx::rewriteModel(m);
      check(s.rewrites==1,"valid mean was not rewritten");check(m.graph().node(0).op_type()=="Conv","no Conv");
      check(m.graph().node(0).doc_string()=="preserve me","lost metadata");
      check(m.graph().node(0).output(0)=="y","lost output name");
      auto& kernel=m.graph().node(0).attribute(0);
      check(kernel.name()=="kernel_shape" && kernel.ints_size()==2 &&
            kernel.ints(0)==1 && kernel.ints(1)==1,"missing explicit pointwise kernel");
      auto& w=m.graph().initializer(m.graph().initializer_size()-1);
      check(w.dims_size()==4 && w.dims(1)==3 && w.float_data_size()==3,"wrong weights");
      check(std::abs(w.float_data(0)-1.0f/3)<1e-7,"wrong mean coefficient");
      auto stable=m.SerializeAsString();check(vbx::rewriteModel(m).rewrites==0,"not idempotent");
      check(stable==m.SerializeAsString(),"idempotent pass changed bytes");
    }
    sameWhenSkipped(fixture(2));sameWhenSkipped(fixture(1,17,0));sameWhenSkipped(fixture(1,22));
    auto m=fixture();m.mutable_graph()->mutable_input(0)->mutable_type()->mutable_tensor_type()->set_elem_type(onnx::TensorProto::INT8);sameWhenSkipped(m);
    m=fixture();auto* dim=m.mutable_graph()->mutable_input(0)->mutable_type()->mutable_tensor_type()->mutable_shape()->mutable_dim(1);
    dim->clear_dim_value();dim->set_dim_param("C");sameWhenSkipped(m);
    m=fixture();m.mutable_graph()->mutable_node(0)->set_domain("custom");sameWhenSkipped(m);
    m=fixture();m.mutable_graph()->mutable_input(0)->set_name("__vbx_mean_0_weight");m.mutable_graph()->mutable_node(0)->set_input(0,"__vbx_mean_0_weight");
    check(vbx::rewriteModel(m).rewrites==1,"collision case failed");
    check(m.graph().node(0).input(1)!="__vbx_mean_0_weight","name collision");
    m=fixture(1,18);auto* t=m.mutable_graph()->mutable_initializer(0);t->clear_int64_data();t->set_raw_data(std::string("\x01\0\0\0\0\0\0\0",8));
    check(vbx::rewriteModel(m).rewrites==1,"raw axes failed");
    m=fixture(1,18);info(m.mutable_graph()->add_input(),"axes",{1},onnx::TensorProto::INT64);sameWhenSkipped(m);
    m=fixture();m.mutable_graph()->mutable_node(0)->set_input(0,"missing");bool failed=false;
    try{vbx::rewriteModel(m);}catch(const std::runtime_error&){failed=true;}check(failed,"undefined input accepted");
    std::cout<<"rewrite core: all checks passed\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
