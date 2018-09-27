import tvm


def print_schedule(sched, arglist):
    print(tvm.lower(sched, arglist, simple_mode=True))

    
rows = tvm.var("rows")
cols = tvm.var("cols")
max_chans = tvm.const(5)
chans = tvm.var("chans")

input_vec = tvm.placeholder((rows,cols,chans), dtype="float32")
kernel = tvm.compute((cols,chans)
                     , lambda c, cc: 1.0 * c * cc
                     , name="kern_vec")

result = tvm.compute((rows,cols,chans)
                     , lambda y, x, c: input_vec[y,x,c] * kernel[x, c]
                     , name="answer")

sched = tvm.create_schedule(result.op)
result_stage = sched[result]
kernel_stage = sched[kernel]

arglist=[input_vec,result]

print_schedule(sched, arglist)

[co,ci] = result_stage.split(result.op.axis[2], 5)

kernel_stage.compute_at(result_stage, co)

print_schedule(sched, arglist)

result_stage.bind(result.op.axis[0], tvm.thread_axis("blockIdx.x"))
result_stage.bind(result.op.axis[1], tvm.thread_axis("threadIdx.x"))



fun = tvm.build(sched, arglist, "opencl", name="test_compute_at")
