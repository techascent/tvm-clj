# An Introduction To The Tensor Virtual Machine


The Tensor Virtual Machine, or [TVM](https://github.com/dmlc/tvm) is a system for dynamically
generating extremely high performance numeric code.  Today we will be using
[clojure](https://github.com/tech-ascent/tvm-clj) bindings to explore what exactly is
going on.


We will be using the foundation from [datatype](http://techascent.com/blog/datatype-library.html), 
[native-pointers](http://techascent.com/blog/native-pointers.html), 
[opencv](http://techascent.com/blog/opencv-love.html), 
the [tech.compute](https://github.com/tech-ascent/tech.compute) library, and 
[high-performance-compilers](http://techascent.com/blog/high-performance-compilers.html).  
If you haven't read those then the magic behind the scenes make sense but we are think this 
post is still digestable without those sections.


Our motivating example is a 3x3 box blur algorithm.  This is a simple enough algorithm
that still has some significant room for interesting optimization.  Let's walk through
building out the operation.


## Problem Setup


We will be walking through demonstraction code:
[box_blur.clj](src/box_blur.clj).  A dockerfile is provided along with scripts
so those of you on a Mac can walk through it also.  The current tvm-clj bindings do not
include Mac binaries as I have no mac.  You can, however, as mac or windows, build
tvm and install it.  As long as it is on the path the jna loader should pick it up
before attempting to unpack jar resources.


```clojure
(ns box-blur
  (:require [tvm-clj.api :as api]
            [tech.opencv :as opencv]
            [tech.datatype.base :as dtype]
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
;;Note we aren't warning on boxed math.  The small amounts of boxed
;;math that happen in this file are irrelevant for speed purposes; its
;;all compiled to the specific backend.

;;A 3x3 box blur.  Using the api sugar liberally to make the code shorter
;;and more readable.
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

* `lambda`: A 'fn' but with metadata describing the argument list.
* `tvar`: A scalar variable.
* `tlet`: Not used, but you can use 'let' statements in the tensor code.

The algorithm creates a padded image, then an image blurred only in X, finally a fully
blurred image and then casts the result back into the base uint8 datatype.  Blurring
reduces the blurred dimension by 2 pixels as the 3x3 kernel hits the edge pixel.  The
input is clamped-to-edge meaning edge pixels are repeated.


A compute operation has a number of 'members':  Input tensors, output tensors, iteration
axis, etc.  The output tensors refer back to the generating operation via their 'op' member.


What we are doing here is describing, via an AST, the algorithm we are talking about.


```clojure

(defn time-schedule
  [schedule-fn & {:keys [device-type]
                  :or {device-type :cpu}}]
  (let [driver (tvm-reg/get-driver device-type)]
    (first
     (vf/tensor-context
      driver
      :uint8
      (let [src-img (opencv/load "test/data/test.jpg")
            src-tensor (tvm-tm/typed-pointer->tensor src-img)
            [src-height src-width src-chan] (m/shape src-img)
            dst-img (opencv/new-mat src-height src-width src-chan
                                    :dtype :uint8)
            dst-tensor (tvm-tm/typed-pointer->tensor dst-img)
            ;;Try changing the intermediate datatype
            {:keys [arglist schedule bind-map]} (-> (box-blur-fn :uint16)
                                                    (schedule-fn device-type))
            _ (println (api/schedule->str schedule arglist "box_blur"))
            box-blur (tvm-reg/schedule->fn driver {:schedule schedule
                                                   :arglist arglist
                                                   :name :blox-blur
                                                   :bind-map (or bind-map {})})]
        ;;warmup
        (box-blur src-tensor dst-tensor)

        _ (when-not (= :cpu device-type)
            (drv/copy-device->host ct/*stream*
                                   (ct/tensor->buffer dst-tensor) 0
                                   (cpu/ptr->device-buffer dst-img) 0
                                   (m/ecount dst-tensor))
            (drv/sync-with-host ct/*stream*))
        (let [time-result
              (with-out-str
                (time
                 (do
                   (dotimes [iter 20]
                     (box-blur src-tensor dst-tensor))
                   (drv/sync-with-host ct/*stream*))))]

          (opencv/save dst-img "result.jpg")
          time-result))))))
```

Here we have defined a way to time schedules given a source image.  We build a function
out the combination of the algorithm definition and a schedule, then run it a bit and
see how long it takes.  The conversion from opencv to tensor, in the case of the tvm cpu
backend, is zero-copy.  Else as host buffer is created and uploaded to the device.


### The default schedule

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

The base schedule really just puts all operations at the root and then runs them in
order.  We print out the schedule as that helps understand what is going on.  Note that
the base schedule allocates 2 large temporary buffers and then uses the padded buffer
twice.

Also note that the y-blur step accesses the x-blur result in 3 different y-locations per
output pixel.  This will be important later.


Lets walk through some scheduling operators.  We will not change the box blur function definition.


## A few scheduling primitives.


## Parallelizing

Parallelizing is marking an axis as being safe to operate on in parallel.  This is a
cpu-only instruction as we indicate this information differently to the GPU.


```clojure
(defn parallel
  [{:keys [box-blur y-blur x-blur] :as item} device-type]
  (let [schedule (api/create-schedule [box-blur])
        [y-axis x-axis chan-axis] (:axis box-blur)]
    (api/stage-parallel (schedule box-blur) y-axis)
    (assoc item :schedule schedule)))


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


## Reordering

Change the axis iteration order.


```clojure
(defn reorder
  [{:keys [box-blur y-blur x-blur] :as item} device-type]
  (let [schedule (api/create-schedule [box-blur])
        [x-x-axis x-chan-axis x-y-axis] (:axis x-blur)
        [y-axis x-axis chan-axis] (:axis box-blur)]
    (api/stage-reorder (schedule x-blur) [x-y-axis x-x-axis x-chan-axis])
    (assoc item :schedule schedule)))

blog.tvm.box-blur> (time-schedule reorder)
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
  for (c, 0, 3) {
    for (y, 0, (height + 2)) {
      for (x, 0, width) {
        x_blur[((c + ((y*width)*3)) + (x*3))] = (((padded[((c + ((y*(width + 2))*3)) + (x*3))] + padded[(((c + ((y*(width + 2))*3)) + (x*3)) + 3)]) + padded[(((c + ((y*(width + 2))*3)) + (x*3)) + 6)])/(uint16)3)
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

"Elapsed time: 57.641104 msecs"
```


### Tiling

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


blog.tvm.box-blur> (time-schedule tiled-schedule)
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
produce box_blur {
  for (y, 0, height) {
    for (x, 0, width) {
      for (c, 0, 3) {
        buffer[((((y*width) + x)*3) + c)] = uint8((float32(padded[((((y*width) + x)*3) + c)]) + 0.500000f))
      }
    }
  }
}

"Elapsed time: 44.106099 msecs"

blog.tvm.box-blur>
```

Tiling is operating over sub-regions of our iteration space.  It is equivalent to:

```clojure
(let [[y-outer y-inner] (api/stage-split-axis y-blur-stage y-blur-y-axis 16)
      [x-outer x-inner] (api/stage-split-axis y-blur-stage y-blur-x-axis 3)]
  (api/stage-reorder y-blur-stage [y-outer x-outer y-inner x-inner]))
```

### Inlining

Inline the computation of the result into the next operation.


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


### Compute At

Compute the given partial result at a given axis of another operation:


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


## An Optimized Schedule


Of course I spent some time and attempted to create a schedule that is pretty
quick.  I was unable to match opencv in this case, however.  I think most likely I am
missing access to some vector instructions that work particularly well for this case.
[Here](https://github.com/tech-ascent/tvm-clj/blob/master/README.md#image-scaling-tvm-vs-opencv)
I have implementations of area and bilinear resizing of image that do outperform their
opencv counterparts.  


One thing to note is that the above algorithm would extend easily
to an arbitrary number of channels *and* and arbitrary base datatypes.  This is useful,
for instance, in satellite imagery where you can have far more than 3 channels and you may
have also unsigned short datatypes.  I do not believe this to be true of the sse-optimized versions
present in openCV.




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


For reference, here is opencv doing same thing:

```clojure
blog.tvm.box-blur> (time-opencv)
"Elapsed time: 5.845806 msecs"
```


## Generalized GPU Support!!


TVM comes with many backends - cpu (llvm), cuda, opencl, vulkan...and many more!  We
have setup our structure carefully to allow opencl to work identically.  In this case
we cannot do zero-copy transfer to/from the device.  Note this all builds on the
compute-device framework.  This test was done on a machine with an NVIDIA 1070 GPU.

In a laptop with an integrated GPU you would not expect to see such improvements but
hey, you can still at least use opencl to offload some work.


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


## What just happened?


Original Image: ![original](test/data/test.jpg)


Blurred Image: ![blurred](images/result-cpu.jpg)


1.  We defined an algorithm.
1.  We scheduled it many different ways.
1.  We built a highly optimized native function from the clojure repl (!!).
1.  We built a highly optimized opencl function from the clojure repl (!!).


We enabled generalized high performance numeric programming in clojure with compiler
that is capable of state of the art performance.


## What this means to me


I am really happy to bring you this.  For me this is a great thing to see a field that
was so extremely difficult and arcane (high performance computing) start to open up with
more general and powerful tools.  I have been a part of GPGPU programming since it
started it has always required quite specialized knowledge along with extensive and
error prone re-write of algorithms.


It is clear that high performance numeric programming is very important to the future of
computing.  It is also clear that high performance architectures will continue to change
and their complexity, especially in a programmatic sense, will never decrease
substantially.  A codebase described as above will be much faster to adapt to new
architectures.


One thing that Rich consistently points out is that maintaining programs over a long period
of time is difficult and requires them to change in ways the original designers will have
never considered.  Hardware architectures also change and thus the ability to
adapt to new ones is a large and important component to maintaining programs over time if
those programs require high performance numerics.


I have used clojure since 2008.  I enjoy it and gain the most leverage from it
compared to any language that I know.  Access to a compiler, infrastructure, and ecosystem like TVM 
enables me to attack a class of problems using clojure that I was previously unable to seriously consider.
