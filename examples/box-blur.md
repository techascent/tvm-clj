# TVM Time


The Tensor Virtual Machine, or [TVM](https://github.com/dmlc/tvm) is a system
for dynamically generating high performance numeric code. We have developed
[Clojure bindings to TVM](https://github.com/tech-ascent/tvm-clj), and this post
will employ them to explore some of what is possible.


This post builds on our earlier posts:
[datatype](http://techascent.com/blog/datatype-library.html),
[native-pointers](http://techascent.com/blog/native-pointers.html),
[OpenCV](http://techascent.com/blog/opencv-love.html), the
[tech.compute](https://github.com/tech-ascent/tech.compute) library, and
[high-performance-compilers](http://techascent.com/blog/high-performance-compilers.html).
Those posts cover many behind-the-scenes details, but the aim is to have this
remain digestible on its own.

Our motivating example is a 3x3 box blur. This classic example is a simple
enough algorithm, but still has some significant room for interesting
optimization. Let's walk through building out the operation.

#### Problem Setup

Here is a link to the full demonstration code:
[box_blur.clj](https://github.com/tech-ascent/tvm-clj/blob/master/examples/src/box_blur.clj).
A Dockerfile is provided along with scripts so anyone can follow along,
regardless of their operating system or other hardware details. The current
`tvm-clj` bindings do not include Mac or Windows binaries, get those from TVM
yourself. As long as TVM is on your path, the JNA loader should pick it up
before attempting to unpack jar resources. TVM binds tightly to the system it is
built on (CUDA, CUDNN, MKL, ROCM, Vulkan) etc so having the system provide it
makes more sense than putting it in a jar.

##### The Algorithm

Our strategy has three steps:

1. Create a padded image
2. Blur only in `X`
3. Blur in `Y`

Blurring reduces each blurred dimension by 2 pixels as the 3x3 kernel hits an
edge pixel. We choose to make the input clamped-to-edge, meaning edge pixels are
repeated.

The following function doesn't actually do any computation, instead it produces
an AST we then later apply schedules to.

```clojure
(ns box-blur
  (:require [tvm-clj.api :as api]
            [tech.opencv :as opencv]
            [tech.datatype :as dtype]
            [tech.compute.tvm :as tvm]
            [tech.compute.verify.tensor :as vf]
            [tech.compute :as compute]
            [clojure.core.matrix :as m]
            [tech.compute.tensor :as ct]
            [tech.compute.tensor.defaults :as ct-defaults]
            [tvm-clj.api-sugar :refer :all]
            [tech.compute.tvm.cpu]
            [tech.compute.tvm.gpu]
            [tech.compute.tvm.tensor-math])
  (:refer-clojure :exclude [+ - / * cast min max = rem])
  (:import [org.bytedeco.javacpp opencv_imgproc]))

(set! *warn-on-reflection* true)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Note we aren't warning on boxed math.  The small amounts of boxed math that
;; happen in this file are irrelevant for speed purposes; the interesting bits
;; of computation are all compiled and run on the specific backend.

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;A 3x3 box blur. With liberal api sugar to make the code shorter and readable.
(defn box-blur-fn
  [intermediate-datatype]
  (let [in-height (tvar "height")
        in-width (tvar "width")
        in-channels (int 3)
        input (placeholder [in-height in-width in-channels] "input"
                           :dtype :uint8)

        clamp (fn [val val_min val_max]
                (-> (min val val_max)
                    (max val_min)))
        ;;Read a clamped-to-edge value of the source image
        read-element (fn [img y x c]
                       (img [(clamp y 0 (- in-height 1))
                             (clamp x 0 (- in-width 1))
                             c]))
        padded-input (compute
                      [(+ in-height 2) (+ in-width 2) in-channels]
                      (lambda [y x c]
                              (-> (read-element input (- y 1) (- x 1) c)
                                  (cast intermediate-datatype)))
                      "padded")
        x-blur-vec (compute
                    [(+ in-height 2) in-width in-channels]
                    (lambda
                     [y x c]
                     (+ (padded-input [y x c])
                        (padded-input [y (+ x 1) c])
                        (padded-input [y (+ x 2) c])))
                    "x-blur")
        y-blur-vec (compute
                    [in-height in-width in-channels]
                    (lambda [y x c]
                            (+ (x-blur-vec [y x c])
                               (x-blur-vec [(+ y 1) x c])
                               (x-blur-vec [(+ y 2) x c])))
                    "y-blur")
        output (compute
                [in-height in-width in-channels]
                (lambda [y x c]
                        (-> (y-blur-vec [y x c])
                            (/ (const 9 intermediate-datatype))
                            (cast :uint8)))
                "box-blur")]
    {:input input
     :padded padded-input
     :x-blur x-blur-vec
     :y-blur y-blur-vec
     :box-blur output
     :output output
     :in-height in-height
     :in-width in-width
     :in-channels in-channels
     :arglist [input output]}))
```

Some important concepts that are probably new here:

* `lambda`: A 'fn' but with metadata describing the argument list.
* `tvar`: A scalar variable.
* `tlet`: Not used, but you can use 'let' statements in the tensor code.

##### Scheduling, Compiling, Running

Here we define a way to time different schedules given a source image. First we
build a function out the combination of the algorithm definition and a
schedule, then run it and see how long it takes. Notably, the conversion from
OpenCV to tensor, in the case of the tvm CPU backend, is zero-copy. In other
cases a host buffer is created and uploaded to the device.

```clojure
(defn time-schedule
  [schedule-fn & {:keys [device-type algorithm-fn]
                  :or {device-type :cpu
                       ;;Could also try no-sugar
                       algorithm-fn box-blur-fn}}]
  (let [driver (tvm/driver device-type)]
    ;;Bind the default compute stream for the default device for this driver
    ;;Bind the default datatype to use if non are provided.
    (vf/tensor-default-context
     driver
     :uint8

     (let [src-img (opencv/load "test/data/test.jpg")
           src-tensor (if (= :cpu device-type)
                        ;;If we are a cpu device then we can use the opencv image
                        ;;directly as a tensor.
                        src-img
                        ;;Else we upload the image to the device returning a new
                        ;;tensor that has a buffer on the device.
                        ct/clone-to-device)
           ;;opencv images implement tech.datatype.base/PPrototype
           ;;We can create on like this one or we can clone exactly this one.
           dst-img (dtype/from-prototype src-img)

           ;;A terse way of stating the if condition above.  cond-> threads the first
           ;;argument through the clauses that are true and then returns the result.
           dst-tensor (cond-> dst-img
                        (not= :cpu device-type)
                        ct/clone-to-device)

           ;;Call the algorithm-fn.  This generates an AST that describes our algorithm
           ;;Then schedule it on the given device type.
           {:keys [arglist schedule bind-map]} (-> (algorithm-fn :uint16)
                                                   (schedule-fn device-type))
           ;;Always helpful
           _ (println (api/schedule->str schedule arglist "box_blur"))
           ;;Schedule the function and return a normal clojure function that will do the
           ;;thing.
           box-blur (tvm/schedule->fn driver {:schedule schedule
                                              :arglist arglist
                                              :name :blox-blur
                                              :bind-map (or bind-map {})})]

       ;;Note that on the cpu, the OpenCV image itself is used with no copy nor
       ;;transformation.  TVM can operate directly on items that implement enough
       ;;protocols: (tech.compute.tvm.driver/acceptable-tvm-device-buffer?).
       ;;This is the 'warmup' run.
        (box-blur src-tensor dst-tensor)

        _ (when-not (= :cpu device-type)
            ;;If we aren't on the cpu then transfer the data back to the cpu.
            (ct/assign! dst-img dst-tensor))

        (let [time-result
              (simple-time
               (box-blur src-tensor dst-tensor)
               ;;We need to sync with the host in order to fairly time the algorithm.
               ;;Without this the algorithm continues to run in the background but the
               ;;save immediately below would write indeterminate results to the disk.
               (compute/sync-with-host (ct-defaults/infer-stream {})))]

          ;;Write the result and off we go!
          (opencv/save dst-img (format "result-%s.jpg" (name device-type)))
          time-result)))))
```


##### The default schedule

Here we define the base schedule, which simply puts all operations at the root
and then runs them in order. We print out the schedule in order to understand
what is going on. Note that the base schedule allocates 2 large temporary
buffers and then uses the padded buffer twice.

Also note that the y-blur step accesses the x-blur result in 3 different
y-locations per output pixel.

```clojure
(defn base-schedule
  [item device-type]
  (let [target (:box-blur item)
        schedule (api/create-schedule [target])]
    (assoc item :schedule schedule
           :arglist [(:input item) target]
           :output target)))


blog.tvm.box-blur> (time-schedule base-schedule)
// attr [padded] storage_scope = "global"
allocate padded[uint16 * (max(((height + 2)*(width + 2)), (height*width))*3)]
// attr [x_blur] storage_scope = "global"
allocate x_blur[uint16 * (height + 2) * width * 3]
produce padded {
  for (y, 0, (height + 2)) {
    for (x, 0, (width + 2)) {
      for (c, 0, 3) {
        padded[((((y*(width + 2)) + x)*3) + c)] = uint16(buffer[(((max((min(x, width) + -1), 0) + (max((min(y, height) + -1), 0)*width))*3) + c)])
      }
    }
  }
}
produce x_blur {
  for (y, 0, (height + 2)) {
    for (x, 0, width) {
      for (c, 0, 3) {
        x_blur[((((y*width) + x)*3) + c)] = (((padded[(((((y*(width + 2)) + x)*3) + c) + 3)] + padded[(((((y*(width + 2)) + x)*3) + c) + 6)]) + padded[((((y*(width + 2)) + x)*3) + c)])/(uint16)3)
      }
    }
  }
}
produce y_blur {
  for (y, 0, height) {
    for (x, 0, width) {
      for (c, 0, 3) {
        padded[((((y*width) + x)*3) + c)] = (((x_blur[((((y*width) + x)*3) + c)] + x_blur[(((((y + 1)*width) + x)*3) + c)]) + x_blur[(((((y + 2)*width) + x)*3) + c)])/(uint16)3)
      }
    }
  }
}
produce box_blur {
  for (y, 0, height) {
    for (x, 0, width) {
      for (c, 0, 3) {
        buffer[((((y*width) + x)*3) + c)] = uint8((float32(padded[((((y*width) + x)*3) + c)]) + 0.500000f))
      }
    }
  }
}

"Elapsed time: 51.458027 msecs"

```

So, that is the baseline. Variations in scheduling allow for some optimization.

#### A Few Scheduling Primitives

##### Parallelizing

Parallelizing is marking an axis as being safe to operate on in parallel. This
is a CPU-only instruction, and is done differently on GPU.

```clojure
(defn parallel
  [{:keys [box-blur y-blur x-blur] :as item} device-type]
  (let [schedule (api/create-schedule [box-blur])
        [y-axis x-axis chan-axis] (:axis box-blur)]
    (api/stage-parallel (schedule box-blur) y-axis)
    (assoc item :schedule schedule)))


produce box_blur {
  parallel (y, 0, height) {
    for (x, 0, width) {
      for (c, 0, 3) {
        buffer[((((y*width) + x)*3) + c)] = uint8((float32(padded[((((y*width) + x)*3) + c)]) + 0.500000f))
      }
    }
  }
}

"Elapsed time: 40.606078 msecs"
```

OK, faster.

##### Reordering

Here we change the axis iteration order.

```clojure
(defn reorder
  [{:keys [box-blur y-blur x-blur] :as item} device-type]
  (let [schedule (api/create-schedule [box-blur])
        [x-x-axis x-chan-axis x-y-axis] (:axis x-blur)
        [y-axis x-axis chan-axis] (:axis box-blur)]
    (api/stage-reorder (schedule x-blur) [x-y-axis x-x-axis x-chan-axis])
    (assoc item :schedule schedule)))


produce x_blur {
  for (c, 0, 3) {
    for (y, 0, (height + 2)) {
      for (x, 0, width) {
        x_blur[((c + ((y*width)*3)) + (x*3))] = (((padded[((c + ((y*(width + 2))*3)) + (x*3))] + padded[(((c + ((y*(width + 2))*3)) + (x*3)) + 3)]) + padded[(((c + ((y*(width + 2))*3)) + (x*3)) + 6)])/(uint16)3)
      }
    }
  }
}

"Elapsed time: 57.641104 msecs"
```

Hmm...

##### Split Axis

Split an axis up into [outer inner] domain variables:

```clojure
(defn split-axis
  [{:keys [box-blur y-blur x-blur] :as item} device-type]
  (let [schedule (api/create-schedule [box-blur])
        [x-x-axis x-chan-axis x-y-axis] (:axis x-blur)
        [y-axis x-axis chan-axis] (:axis box-blur)]
    (api/stage-split-axis (schedule box-blur) y-axis 16)
    (assoc item :schedule schedule)))

produce box_blur {
  for (y.outer, 0, ((height + 15)/16)) {
    for (y.inner, 0, 16) {
      for (x, 0, width) {
        for (c, 0, 3) {
          if (likely(((y.outer*16) < (height - y.inner)))) {
            buffer[((((((y.outer*16) + y.inner)*width) + x)*3) + c)] = uint8((padded[((((((y.outer*16) + y.inner)*width) + x)*3) + c)]/(uint16)9))
          }
        }
      }
    }
  }
}

top-3 times: 41.44ms 41.67ms 41.90ms
```

##### Tiling

Tiling is operating over sub-regions of our iteration space. It is equivalent to:

```clojure
(let [[y-outer y-inner] (api/stage-split-axis y-blur-stage y-blur-y-axis 16)
      [x-outer x-inner] (api/stage-split-axis y-blur-stage y-blur-x-axis 3)]
  (api/stage-reorder y-blur-stage [y-outer x-outer y-inner x-inner]))
```

```clojure
(defn tiled-schedule
  [{:keys [box-blur y-blur] :as item} device-type]
  (let [schedule (api/create-schedule [box-blur])
        [y-axis x-axis chan-axis] (:axis y-blur)
        tile-axis
        (api/stage-tile (schedule y-blur)
                        y-axis x-axis 16 3)]
    (assoc item
           :schedule schedule
           :tile-axis tile-axis)))

produce y_blur {
  for (y.outer, 0, ((height + 15)/16)) {
    for (x.outer, 0, ((width + 2)/3)) {
      for (y.inner, 0, 16) {
        for (x.inner, 0, 3) {
          for (c, 0, 3) {
            if (likely(((y.outer*16) < (height - y.inner)))) {
              if (likely(((x.outer*3) < (width - x.inner)))) {
                padded[(((((x.outer*3) + (((y.outer*16) + y.inner)*width)) + x.inner)*3) + c)] = (((x_blur[(((((x.outer*3) + ((((y.outer*16) + y.inner) + 1)*width)) + x.inner)*3) + c)] + x_blur[(((((x.outer*3) + ((((y.outer*16) + y.inner) + 2)*width)) + x.inner)*3) + c)]) + x_blur[(((((x.outer*3) + (((y.outer*16) + y.inner)*width)) + x.inner)*3) + c)])/(uint16)3)
              }
            }
          }
        }
      }
    }
  }
}

"Elapsed time: 44.106099 msecs"

blog.tvm.box-blur>
```

##### Inlining

It is possible to inline the computation of the result into the next operation.


```clojure
(defn inline
  [{:keys [padded x-blur y-blur box-blur] :as item} device-type]
  (let [schedule (api/create-schedule [box-blur])]
    (api/stage-inline (schedule padded))
    (api/stage-inline (schedule x-blur))
    (api/stage-inline (schedule y-blur))
    (assoc item :schedule schedule)))


blog.tvm.box-blur> (time-schedule inline)
produce box_blur {
  for (y, 0, height) {
    for (x, 0, width) {
      for (c, 0, 3) {
        buffer[((((y*width) + x)*3) + c)] = uint8((float32(((((((uint16(buffer[(((max((min(x, width) + -1), 0) + (max((min(y, height) + -1), 0)*width))*3) + c)]) + uint16(buffer[(((max(min(x, (width + -1)), 0) + (max((min(y, height) + -1), 0)*width))*3) + c)])) + uint16(buffer[(((max(min((x + 1), (width + -1)), 0) + (max((min(y, height) + -1), 0)*width))*3) + c)]))/(uint16)3) + (((uint16(buffer[(((max((min(x, width) + -1), 0) + (max(min(y, (height + -1)), 0)*width))*3) + c)]) + uint16(buffer[(((max(min(x, (width + -1)), 0) + (max(min(y, (height + -1)), 0)*width))*3) + c)])) + uint16(buffer[(((max(min((x + 1), (width + -1)), 0) + (max(min(y, (height + -1)), 0)*width))*3) + c)]))/(uint16)3)) + (((uint16(buffer[(((max((min(x, width) + -1), 0) + (max(min((y + 1), (height + -1)), 0)*width))*3) + c)]) + uint16(buffer[(((max(min(x, (width + -1)), 0) + (max(min((y + 1), (height + -1)), 0)*width))*3) + c)])) + uint16(buffer[(((max(min((x + 1), (width + -1)), 0) + (max(min((y + 1), (height + -1)), 0)*width))*3) + c)]))/(uint16)3))/(uint16)3)) + 0.500000f))
      }
    }
  }
}

"Elapsed time: 39.468657 msecs"
```

#### Compute At

In at least some cases, these ideas can be usefully composed.

Here we compute a partial result at a given axis of another operation:

```clojure
(defn compute-at
  [{:keys [padded x-blur y-blur box-blur] :as item} device-type]
  (let [schedule (api/create-schedule [box-blur])
        [y-axis x-axis chan-axis] (:axis box-blur)]
    (api/stage-inline (schedule x-blur))
    (api/stage-inline (schedule padded))
    (api/stage-compute-at (schedule y-blur)
                          (schedule box-blur) y-axis)
    (assoc item :schedule schedule)))


blog.tvm.box-blur> (time-schedule compute-at)
// attr [y_blur] storage_scope = "global"
allocate y_blur[uint16 * 1 * width * 3]
produce box_blur {
  for (y, 0, height) {
    produce y_blur {
      for (x, 0, width) {
        for (c, 0, 3) {
          y_blur[((x*3) + c)] = ((((((uint16(buffer[(((max((min(x, width) + -1), 0) + (max((min(y, height) + -1), 0)*width))*3) + c)]) + uint16(buffer[(((max(min(x, (width + -1)), 0) + (max((min(y, height) + -1), 0)*width))*3) + c)])) + uint16(buffer[(((max(min((x + 1), (width + -1)), 0) + (max((min(y, height) + -1), 0)*width))*3) + c)]))/(uint16)3) + (((uint16(buffer[(((max((min(x, width) + -1), 0) + (max(min(y, (height + -1)), 0)*width))*3) + c)]) + uint16(buffer[(((max(min(x, (width + -1)), 0) + (max(min(y, (height + -1)), 0)*width))*3) + c)])) + uint16(buffer[(((max(min((x + 1), (width + -1)), 0) + (max(min(y, (height + -1)), 0)*width))*3) + c)]))/(uint16)3)) + (((uint16(buffer[(((max((min(x, width) + -1), 0) + (max(min((y + 1), (height + -1)), 0)*width))*3) + c)]) + uint16(buffer[(((max(min(x, (width + -1)), 0) + (max(min((y + 1), (height + -1)), 0)*width))*3) + c)])) + uint16(buffer[(((max(min((x + 1), (width + -1)), 0) + (max(min((y + 1), (height + -1)), 0)*width))*3) + c)]))/(uint16)3))/(uint16)3)
        }
      }
    }
    for (x, 0, width) {
      for (c, 0, 3) {
        buffer[((((y*width) + x)*3) + c)] = uint8((float32(y_blur[((x*3) + c)]) + 0.500000f))
      }
    }
  }
}

"Elapsed time: 38.438251 msecs"
```


#### An Optimized Schedule


Different attempts can be made to create schedules, and some are faster than
others. However, nothing we tried during the preparation of this post was as
fast as OpenCV, on the CPU, in this case. Chances are, they use a special vector
instruction that works especially well for this particular case. Though, for
other
[algorithms](https://github.com/tech-ascent/tvm-clj/blob/master/README.md#image-scaling-tvm-vs-opencv),
(e.g., area and bi-linear image resize) we can outperform their OpenCV
counterparts.

Impressively, the above algorithm would extend easily to an arbitrary number of
channels *and* arbitrary base datatypes. This is useful, for instance, in
satellite imagery where one often encounters far more than 3 channels, and
unusual datatypes like `unsigned short`. In contrast, other systems like the
SSE-optimized algorithms in OpenCV encounter difficulty here; if adaption is
possible at all.

Now we employ all the toys:

```clojure
(defn all-the-toys
  [{:keys [padded box-blur x-blur y-blur in-height in-width in-channels]
    :as item} device-type]
  (let [schedule (api/create-schedule [box-blur])
        [x-blur-y-axis x-blur-x-axis x-blur-chan-axis] (:axis x-blur)
        [y-axis x-axis chan-axis] (:axis box-blur)
        [y-blur-y-axis y-blur-x-axis y-blur-chan-axis] (:axis y-blur)
        [y-outer x-outer y-inner x-inner] (api/stage-tile
                                           (schedule box-blur)
                                           y-axis x-axis
                                           8 8)
        ;;Here I had to ask some questions:
        ;;
        ;;https://discuss.tvm.ai/t/how-to-control-data-layout-of-intermediate-values/898/4

        ;;We want to calculate X in xyc order but store in xcy order as that is the
        ;;order it is accessed.  This requires setting up a cache operation.
        ;;First, reorder X so that the computation happens in xcy order.  Note that this
        ;;will make calculating X inefficient at this point due to the extremely out-of-order access of
        ;;the padded input:.  It will make the y-blur more efficient, however.
        _ (api/stage-reorder (schedule x-blur) [x-blur-x-axis x-blur-chan-axis x-blur-y-axis])
        {x-blur-cache :tensor
         schedule :schedule} (api/schedule-cache-write schedule x-blur "local")
        [cache-x-axis cache-chan-axis cache-y-axis] (:axis x-blur-cache)]
    (api/stage-inline (schedule padded))
    (api/stage-inline (schedule x-blur))
    (api/stage-inline (schedule y-blur))
    ;;Reorder the cache stage to compute in y,x,c order.  This means we will read input in a sane order but
    ;;write to the cache out of order.  This is fine because we aren't using those writes for a while meaning
    ;;the memory stalls will be ignored till much later.  Since we are writing thread-local only the system may
    ;;not even write the cache back out to main memory saving a ton of time.
    (api/stage-reorder (schedule x-blur-cache) [cache-y-axis cache-x-axis cache-chan-axis])

    (if (= device-type :cpu)
      (do
        ;;schedule the cache operation to happen
        (api/stage-compute-at (schedule x-blur-cache)
                              (schedule box-blur) x-outer)
        (api/stage-parallel (schedule box-blur) y-outer))
      (let [gpu-thread-axis (api/stage-fuse (schedule box-blur)
                                            [x-inner chan-axis])]
        (api/stage-compute-at (schedule x-blur-cache)
                              (schedule box-blur) gpu-thread-axis)
        ;;We lose the cache here really but it does at least run and produce a correct result.
        ;;Really, on a per-block basis, you would want to build the cache as a first step
        ;;using all the threads but that leads to more sophistication than I wanted in a demo.
        ;;For homework, go through the gpu conv layer tvm tutorial and then apply it here.
        (api/stage-bind-gpu (schedule box-blur)
                            [(api/stage-fuse (schedule box-blur)
                                             [y-outer x-outer y-inner])]
                            [gpu-thread-axis])))
    (assoc item :schedule schedule)))


blog.tvm.box-blur> (time-schedule all-the-toys)
produce box_blur {
  parallel (y.outer, 0, ((height + 7)/8)) {
    // attr [x_blur.local] storage_scope = "local"
    allocate x_blur.local[uint16 * 16 * 3 * 10]
    for (x.outer, 0, ((width + 15)/16)) {
      produce x_blur.local {
        for (y.c, 0, 10) {
          for (x.c, 0, 16) {
            for (c.c, 0, 3) {
              if (likely(((x.outer*16) < (width - x.c)))) {
                if (likely(((y.outer*8) < ((height - y.c) + 2)))) {
                  x_blur.local[((y.c + (x.c*30)) + (c.c*10))] = (((uint16(buffer[(((max((min(((x.outer*16) + x.c), width) + -1), 0) + (max((min(((y.outer*8) + y.c), height) + -1), 0)*width))*3) + c.c)]) + uint16(buffer[(((max(min((((x.outer*16) + x.c) + 1), (width + -1)), 0) + (max((min(((y.outer*8) + y.c), height) + -1), 0)*width))*3) + c.c)])) + uint16(buffer[(((max(min(((x.outer*16) + x.c), (width + -1)), 0) + (max((min(((y.outer*8) + y.c), height) + -1), 0)*width))*3) + c.c)]))/(uint16)3)
                }
              }
            }
          }
        }
      }
      for (y.inner, 0, 8) {
        for (x.inner, 0, 16) {
          for (c, 0, 3) {
            if (likely(((y.outer*8) < (height - y.inner)))) {
              if (likely(((x.outer*16) < (width - x.inner)))) {
                buffer[(((((x.outer*16) + (((y.outer*8) + y.inner)*width)) + x.inner)*3) + c)] = uint8((float32((((x_blur.local[((y.inner + (x.inner*30)) + (c*10))] + x_blur.local[(((y.inner + (x.inner*30)) + (c*10)) + 1)]) + x_blur.local[(((y.inner + (x.inner*30)) + (c*10)) + 2)])/(uint16)3)) + 0.500000f))
              }
            }
          }
        }
      }
    }
  }
}

"Elapsed time: 24.955538 msecs"
```

Lowest seen yet.

But, for reference, here is OpenCV doing same thing:

```clojure
blog.tvm.box-blur> (time-opencv)
"Elapsed time: 5.845806 msecs"
```

#### Generalized GPU Support!!

TVM comes with many backends - CPU (LLVM), CUDA, OpenCL, Vulkan...and others.
Everything done so far here can be tried on other backends with no changes. (!)
As expected, working with other hardware does necessitate a discrete transfer
step to and from the device. The `tvm-clj` ecosystem includes a compute-device
framework that abstracts these details in what we think is a nice way.

The following test was done on a machine with an NVIDIA 1070 GPU. On a laptop
with an integrated GPU one would not expect the improvements we see
here. However, you can still at least use OpenCL to offload some work. Again,
without writing any significant low level code.

```clojure
blog.tvm.box-blur> (time-schedule all-the-toys :device-type :opencl)
produce box_blur {
  // attr [iter_var(blockIdx.x, , blockIdx.x)] thread_extent = ((((height + 7)/8)*((width + 15)/16))*8)
  // attr [x_blur.local] storage_scope = "local"
  allocate x_blur.local[uint16 * 1 * 1 * 3]
  // attr [iter_var(threadIdx.x, , threadIdx.x)] thread_extent = 48
  produce x_blur.local {
    for (y.c, 0, 3) {
      if (likely(((((blockIdx.x/8) % ((width + 15)/16))*16) < (width - (threadIdx.x/3))))) {
        if (likely((((0 - y.c) - (blockIdx.x % 8)) <= (((blockIdx.x/8)/((width + 15)/16))*8)))) {
          if (likely(((((blockIdx.x/8)/((width + 15)/16))*8) < (((height - y.c) - (blockIdx.x % 8)) + 2)))) {
            x_blur.local[y.c] = (((uint16(buffer[((max((min(((((blockIdx.x/8) % ((width + 15)/16))*16) + (threadIdx.x/3)), width) + -1), 0)*3) + (((max((min((((((blockIdx.x/8)/((width + 15)/16))*8) + (blockIdx.x % 8)) + y.c), height) + -1), 0)*width)*3) + (threadIdx.x % 3)))]) + uint16(buffer[((max(min((((((blockIdx.x/8) % ((width + 15)/16))*16) + (threadIdx.x/3)) + 1), (width + -1)), 0)*3) + (((max((min((((((blockIdx.x/8)/((width + 15)/16))*8) + (blockIdx.x % 8)) + y.c), height) + -1), 0)*width)*3) + (threadIdx.x % 3)))])) + uint16(buffer[((max(min(((((blockIdx.x/8) % ((width + 15)/16))*16) + (threadIdx.x/3)), (width + -1)), 0)*3) + (((max((min((((((blockIdx.x/8)/((width + 15)/16))*8) + (blockIdx.x % 8)) + y.c), height) + -1), 0)*width)*3) + (threadIdx.x % 3)))]))/(uint16)3)
          }
        }
      }
    }
  }
  if (likely(((((blockIdx.x/8)/((width + 15)/16))*8) < (height - (blockIdx.x % 8))))) {
    if (likely(((0 - (blockIdx.x % 8)) <= (((blockIdx.x/8)/((width + 15)/16))*8)))) {
      if (likely(((((blockIdx.x/8) % ((width + 15)/16))*16) < (width - (threadIdx.x/3))))) {
        buffer[(((((((((blockIdx.x/8)/((width + 15)/16))*8) + (blockIdx.x % 8))*width) + (((blockIdx.x/8) % ((width + 15)/16))*16)) + (threadIdx.x/3))*3) + (threadIdx.x % 3))] = uint8((float32((((x_blur.local[0] + x_blur.local[1]) + x_blur.local[2])/(uint16)3)) + 0.500000f))
      }
    }
  }
}

"Elapsed time: 1.803495 msecs"
```

And there you have it.

#### What Happened?

Original Image: ![original](test/data/test.jpg)

Blurred Image: ![blurred](images/result-cpu.jpg)

1.  We defined an algorithm.
1.  We scheduled it many different ways.
1.  We built a highly optimized native function from the Clojure repl (!!).
1.  We built a highly optimized OpenCL function from the Clojure repl (!!).

This enables generalized high performance numeric programming in Clojure.
Leveraging a compiler that is capable of, and has demonstrated, state of the art
performance.


#### Summary

Clearly, high performance numeric programming is important to the future of
computing. These architectures will continue to change, and programming them
will continue to increase in complexity. Structuring code at as high of a level
as possible enables faster adaptation to coming changes. Having this kind of
access from Clojure maintains the high leverage, and extremely rapid
test/development cycle we have come to love.

Built modules can be saved as dynamically loadable files that TVM can consume
later (potentially from Python or C++). This allows us to participate in the
lively and active TVM community by both producing and consuming TVM artifacts,
all from Clojure.

We use this to:

1. Speed up server tasks without switching languages, saving money on cloud computing costs.
1. Prototype new complex numeric algorithms, doing things that have never been done before.
1. Perform scientific analysis of data previously unavailable in a language as high-level as Clojure.
