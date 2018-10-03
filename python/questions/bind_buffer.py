import tvm


def print_schedule(sched, arglist):
    print(tvm.lower(sched, arglist, simple_mode=True))


rows = tvm.var("rows")
cols = tvm.var("cols")
chans = tvm.var("chans")

input_vec = tvm.placeholder((rows,cols,chans), dtype="float32", name="input")
clamp = lambda v, v_min, v_max: tvm.max( tvm.min(v, v_max), v_min )
## clamp to edge padding
padded = tvm.compute((rows+2,cols+2,chans)
                     , lambda y, x, c: input_vec[clamp(y-1, 0, rows-1)
                                                 , clamp(x-1, 0, cols-1)
                                                 , c].astype("uint16")
                     , name="padded")



x_blur = tvm.compute((rows+2, cols, chans)
                     , lambda y, x, c: (padded[y,x,c] +
                                        padded[y,x+1,c] +
                                        padded[y,x+2,c]) / 3
                     , name="x_blur")

y_blur = tvm.compute((rows, cols, chans)
                     , lambda y, x, c: (x_blur[y,x,c] +
                                        x_blur[y+1,x,c] +
                                        x_blur[y+2,x,c]) / 3
                     , name="y_blur")

box_blur = tvm.compute((rows,cols,chans)
                       , lambda y, x, c: y_blur[y,x,c].astype("uint8")
                       , name="box_blur")

arglist = [input_vec, box_blur]

schedule = tvm.create_schedule(box_blur.op)
schedule[padded.op].compute_inline()
schedule[y_blur].compute_inline()
schedule[x_blur].compute_at(schedule[box_blur], box_blur.op.axis[1])
print_schedule(schedule, arglist)

x_blur_y_stride = 1
x_blur_c_stride = rows + 2
x_blur_x_stride = x_blur_c_stride * 3

fun = tvm.build(schedule, arglist, "llvm", name="box_blur"
                , binds={x_blur: tvm.decl_buffer(x_blur.shape
                                                 , name="x_blur"
                                                 , scope="local"
                                                 , dtype=x_blur.dtype
                                                 , strides=[x_blur_y_stride,
                                                            x_blur_x_stride,
                                                            x_blur_c_stride])})
