package tvm_clj.tvm;


import com.sun.jna.*;

public class CFunction {

  public static interface TVMPackedCFunc extends Callback {
    int invoke(Pointer args, Pointer typeCodes, int numArgs,
	       Pointer retValueHandle, Pointer resourceHandle);
  }

  public static interface TVMPackedCFuncFinalizer extends Callback {
    int invoke(Pointer resourceHandle);
  }
}
