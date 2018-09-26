import tvm


rows = tvm.var("rows")
cols = tvm.var("cols")
chans = tvm.var("chans")
input_vec = tvm.placeholder((rows,cols,chans), dtype="float32")
kernel = tvm.compute((cols,chans)
                     , lambda c, cc: 1.0 * c * cc
                     , name="kern_vec")
result = tvm.compute((rows,cols,chans)
                     , lambda y, x, c: input_vec[y,x,c] * kernel[x,c]
                     , name="answer")

sched = tvm.create_schedule(result.op)
result_stage = sched[result]
kernel_stage = sched[kernel]
kernel_stage.compute_at(result_stage, result.op.axis[1])
result_stage.bind(result.op.axis[0], tvm.thread_axis("blockIdx.x"))
result_stage.bind(result.op.axis[1], tvm.thread_axis("threadIdx.x"))
fun = tvm.build(sched, [input_vec], "opencl", name="test_compute_at")
