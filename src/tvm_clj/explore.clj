(ns tvm-clj.explore
  (:require [libpython-clj.python :as py]
            [tvm-clj.api :as api]))

(py/initialize!)
(def tvm (py/import-module "tvm"))

(def py-p (py/call-attr tvm "placeholder" [2 3]))

(def api-p (api/placeholder [2 3] "jna-p"))

(def ctypes-node (-> (py/get-attr tvm "_ffi")
                     (py/get-attr "_ctypes")
                     (py/get-attr "node")))

(def node-type-map (py/get-attr ctypes-node "NODE_TYPE"))

(def node-type-class (get node-type-map (:tvm-type-index api-p)))

(def node-handle-type (py/get-attr ctypes-node "NodeHandle"))

(def new-handle (let [hdl (py/call-attr node-handle-type "__new__"
                                        node-handle-type)]
                  (py/call-attr hdl "__init__"
                                (com.sun.jna.Pointer/nativeValue
                                 (:tvm-jcpp-handle api-p)))
                  hdl))

(def new-node
  (let [item
        (py/call-attr node-type-class "__new__" node-type-class)]
    (py/set-attr! item "handle" new-handle)
    item))
