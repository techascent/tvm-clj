package tvm_clj.tvm;

import com.sun.jna.*;
import java.util.*;


public interface DLPack extends Library {

  public static final int DLPACK_VERSION = 020;

  public static class DLContext extends Structure {

    public int device_type;
    public int device_id;


    public static class ByReference extends DLContext implements Structure.ByReference {}
    public static class ByValue extends DLContext implements Structure.ByValue {}
    public DLContext () {}
    public DLContext (Pointer p ) { super(p); read(); }
    protected List getFieldOrder() { return Arrays.asList(new String[]
      { "device_type", "device_id" }); }
  }

  public static class DLDataType extends Structure {
    public byte code;
    public byte bits;
    public short lanes;


    public static class ByReference extends DLDataType implements Structure.ByReference {}
    public static class ByValue extends DLDataType implements Structure.ByValue {}
    public DLDataType () {}
    public DLDataType (Pointer p ) { super(p); read(); }
    protected List getFieldOrder() { return Arrays.asList(new String[]
      { "code", "bits", "lanes" }); }
  }

  public static class DLTensor extends Structure {

    public Pointer data;
    public DLContext ctx;
    public int ndim;
    public DLDataType dtype;
    public Pointer shape;
    public Pointer strides;
    public long byte_offset;


    public static class ByReference extends DLTensor implements Structure.ByReference {}
    public static class ByValue extends DLTensor implements Structure.ByValue {}
    public DLTensor () {}
    public DLTensor (Pointer p ) { super(p); read(); }
    protected List getFieldOrder() { return Arrays.asList(new String[]
      { "data", "ctx", "ndim", "dtype", "shape", "strides", "byte_offset"}); }
  }

  public static class DLManagedTensor extends Structure {
    public DLTensor dl_tensor;
    //void* to used to store extra data for deleter
    public Pointer manager_ctx;
    //single argument delete fn, is passed the managed tensor
    public Pointer deleter;


    public static class ByReference extends DLManagedTensor implements Structure.ByReference {}
    public static class ByValue extends DLManagedTensor implements Structure.ByValue {}
    public DLManagedTensor () {}
    public DLManagedTensor (Pointer p ) { super(p); read(); }
    protected List getFieldOrder() { return Arrays.asList(new String[]
      { "dl_tensor", "manager_ctx", "deleter"}); }

  }
}
