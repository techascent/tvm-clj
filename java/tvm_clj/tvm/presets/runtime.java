package tvm_clj.tvm.presets;

import org.bytedeco.javacpp.annotation.*;
import org.bytedeco.javacpp.tools.*;

@Properties(target="tvm_clj.tvm.runtime",
	    value={@Platform(include={"dlpack/dlpack.h", "runtime/c_runtime_api.h", "c_dsl_api.h", "runtime/c_backend_api.h"},
			     includepath={"tvm/include/tvm", "tvm/3rdparty/dlpack/include/"},
			     link="tvm")})

public class runtime implements InfoMapper {
  public void map(InfoMap infoMap) {
    infoMap.put(new Info("__cplusplus","_WIN32").define(false))
      .put(new Info("TVM_DLL").cppTypes().annotations().cppText(""))
      .put(new Info("TVM_WEAK").cppTypes().annotations().cppText(""));
  }
}
