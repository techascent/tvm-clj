(ns tvm-clj.jna.node
  (:require [tvm-clj.jna.base :refer [make-tvm-jna-fn
                                      device-type->int
                                      device-id->int
                                      ptr-ptr
                                      check-call
                                      ->long-ptr
                                      datatype->dl-datatype
                                      int-ptr
                                      long-ptr
                                      global-function
                                      tvm-value->jvm]
             :as jna-base]
            [tvm-clj.bindings.definitions :refer [tvm-datatype->keyword-nothrow
                                                  node-type-name->keyword-map]
             :as bindings-defs]
            [tvm-clj.bindings.protocols :as bindings-proto]
            [tech.jna :refer [checknil] :as jna]
            [tech.resource :as resource]
            [tvm-clj.jna.fns.global :as global-fns])
  (:import [com.sun.jna Native NativeLibrary Pointer Function Platform]
           [com.sun.jna.ptr PointerByReference IntByReference LongByReference]
           [java.util Map List RandomAccess]
           [java.io Writer]
           [tech.v2.datatype ObjectReader ObjectWriter]
           [clojure.lang MapEntry IFn]))



(defmulti get-extended-node-value
  "Override this to enable type-specific lookups into nodes."
  (fn [node-handle item-key]
    (:tvm-type-kwd (:data node-handle))))


(defmethod get-extended-node-value :default
  [& args]
  nil)


(make-tvm-jna-fn TVMNodeListAttrNames
                 "List the node attributes"
                 Integer
                 [node-handle checknil]
                 [out_size int-ptr]
                 [out_array ptr-ptr])


(defn get-node-fields
  [^Pointer handle]
  (let [fields (PointerByReference.)
        num-fields (IntByReference.)]
    (check-call (TVMNodeListAttrNames handle num-fields fields))
    (jna/char-ptr-ptr->string-vec (.getValue num-fields) (.getValue fields))))


(make-tvm-jna-fn TVMNodeGetAttr
                 "Get a node attribute by name"
                 Integer
                 [node-handle checknil]
                 [key jna/string->ptr]
                 [out_value long-ptr]
                 [out_type_code int-ptr]
                 [out_success int-ptr])


(defn tvm-map->jvm
  [tvm-map-node]
  (->> (global-fns/_MapItems tvm-map-node)
       tvm-array->jvm
       (apply hash-map)))


(defn get-node-field
  [^Pointer handle field-name]
  (let [out-tvm-val (LongByReference.)
        out-type-code (IntByReference.)
        out-success (IntByReference.)
        field-name (cond
                     (string? field-name)
                     field-name
                     (keyword? field-name)
                     (name field-name)
                     :else
                     (throw (ex-info "Unrecognized field name type"
                                     {:field-name field-name})))]
    (check-call (TVMNodeGetAttr handle field-name
                                out-tvm-val
                                out-type-code
                                out-success))
    (if (= 1 (.getValue out-success))
      (tvm-value->jvm (.getValue out-tvm-val)
                      (tvm-datatype->keyword-nothrow (.getValue out-type-code)))
      nil)))

(make-tvm-jna-fn TVMNodeGetTypeIndex
                 "Get the type index of a node."
                 Integer
                 [node-hdl checknil]
                 [out_index int-ptr])


(defn get-node-type-index
  [^Pointer handle]
  (let [node-type-data (IntByReference.)]
    (check-call
     (TVMNodeGetTypeIndex handle node-type-data))
    (.getValue node-type-data)))


(make-tvm-jna-fn TVMNodeTypeKey2Index
                 "Convert a type name to a type index."
                 Integer
                 [type_key jna/string->ptr]
                 [out_index int-ptr])


(defn- node-type-name->index
  "Convert a node type name to an index."
  [^String type-name]
  (let [int-data (IntByReference.)]
    (check-call (TVMNodeTypeKey2Index type-name int-data))
    (.getValue int-data)))


(defonce node-type-index->name*
  (delay
   (->> (keys node-type-name->keyword-map)
        (map (fn [type-name]
               [(node-type-name->index type-name) type-name]))
        (into {}))))


(declare construct-node-handle)


(deftype NodeHandle [^Pointer handle fields]
  bindings-proto/PJVMTypeToTVMValue
  (->tvm-value [this] [(Pointer/nativeValue handle) :node-handle])
  bindings-proto/PToTVM
  (->tvm [this] this)
  bindings-proto/PConvertToNode
  (->node [this] this)
  bindings-proto/PTVMNode
  (is-node-handle? [this] true)
  (node-type-index [this] (get-node-type-index handle))
  (node-type-name [this] (get @node-type-index->name*
                              (bindings-proto/node-type-index this)
                              "UnknownTypeName"))
  jna/PToPtr
  (is-jna-ptr-convertible? [item] true)
  (->ptr-backing-store [item] handle)
  Object
  (equals [a b]
    (= (.hashCode a) (.hashCode b)))
  (hashCode [this]
    (-> (global-function "_raw_ptr" this)
        long
        .hashCode))
  (toString [this]
    (jna-base/global-function "_format_str" this))



  Map
  (containsKey [item k] (boolean (contains? fields k)))
  (entrySet [this]
    (->> (.iterator this)
         iterator-seq
         set))
  (get [this obj-key]
    (if (contains? fields obj-key)
      (get-node-field handle (if (string? obj-key)
                               obj-key
                               (name obj-key)))
      (throw (Exception. (format "Failed to get node value: %s" obj-key)))))
  (getOrDefault [item obj-key obj-default-value]
    (if (contains? fields obj-key)
      (.get item obj-key)
      obj-default-value))
  (isEmpty [this] (= 0 (.size this)))
  (keySet [this] fields)
  (size [this] (count fields))
  (values [this] (map #(.get this %) fields))
  Iterable
  (iterator [this]
    (.iterator ^Iterable (map #(MapEntry. % (.get this %)) fields)))
  IFn
  (invoke [this arg] (if (contains? fields arg)
                       (.get this arg)
                       (get-extended-node-value this arg)))
  (applyTo [this arglist]
    (if (= 1 (count arglist))
      (.invoke this (first arglist))
      (throw (Exception. "Too many arguments to () operator")))))


(defmethod print-method NodeHandle
  [hdl w]
  (.write ^Writer w (.toString hdl)))


(defn is-node-handle?
  [item]
  (bindings-proto/is-node-handle? item))


(make-tvm-jna-fn TVMNodeFree
                 "Free a tvm node."
                 Integer
                 [handle checknil])



(deftype ArrayHandle [handle num-items]
  bindings-proto/PJVMTypeToTVMValue
  (->tvm-value [this] [(Pointer/nativeValue handle) :node-handle])
  bindings-proto/PToTVM
  (->tvm [this] this)
  bindings-proto/PConvertToNode
  (->node [this] this)
  bindings-proto/PTVMNode
  (is-node-handle? [this] true)
  (node-type-index [this] (get-node-type-index handle))
  (node-type-name [this] (get @node-type-index->name*
                              (bindings-proto/node-type-index this)
                              "UnknownTypeName"))
  jna/PToPtr
  (is-jna-ptr-convertible? [item] true)
  (->ptr-backing-store [item] handle)
  Object
  (equals [a b]
    (= (.hashCode a) (.hashCode b)))
  (hashCode [this]
    (-> (global-function "_raw_ptr" this)
        long
        .hashCode))
  (toString [this]
    (jna-base/global-function "_format_str" this))

  ObjectReader
  (lsize [this] num-items)
  (read [this idx]
    (global-fns/_ArrayGetItem this idx)))


(defn get-map-items
  [handle]
  (->> (global-fns/_MapItems handle)
       (partition 2)
       (map (fn [[k v]]
              (MapEntry. k v)))))


(deftype MapHandle [handle]
  bindings-proto/PJVMTypeToTVMValue
  (->tvm-value [this] [(Pointer/nativeValue handle) :node-handle])
  bindings-proto/PToTVM
  (->tvm [this] this)
  bindings-proto/PConvertToNode
  (->node [this] this)
  bindings-proto/PTVMNode
  (is-node-handle? [this] true)
  (node-type-index [this] (get-node-type-index handle))
  (node-type-name [this] (get @node-type-index->name*
                              (bindings-proto/node-type-index this)
                              "UnknownTypeName"))
  jna/PToPtr
  (is-jna-ptr-convertible? [item] true)
  (->ptr-backing-store [item] handle)
  Map
  (containsKey [item k] (contains? (.keySet item) (bindings-proto/->node k)))
  (entrySet [this]
    (->> (.iterator this)
         iterator-seq
         set))
  (get [this obj-key]
    (global-fns/_MapGetItem this (bindings-proto/->node obj-key)))
  (getOrDefault [item obj-key obj-default-value]
    (if (contains? item obj-key)
      (.get item obj-key)
      obj-default-value))
  (isEmpty [this] (= 0 (.size this)))
  (keySet [this] (->> (map first (get-map-items this))
                      set))
  (size [this] (int (global-fns/_MapSize this)))
  (values [this] (map second this))
  Iterable
  (iterator [this]
    (.iterator ^Iterable (get-map-items this)))
  IFn
  (invoke [this arg] (.get this arg))
  (applyTo [this arglist]
    (if (= 1 (count arglist))
      (.invoke this (first arglist))
      (throw (Exception. "Too many arguments to () operator")))))


(defmulti construct-node
  (fn [ptr]
    (-> (NodeHandle. ptr #{})
        (bindings-proto/node-type-name))))


(defmethod construct-node :default
  [ptr]
  (NodeHandle. ptr (->> (get-node-fields ptr)
                        (map keyword)
                        (apply sorted-set))))


(defmethod construct-node "Array"
  [ptr]
  (let [init-handle (NodeHandle. ptr #{})
        node-size (long (global-fns/_ArraySize init-handle))]
    (ArrayHandle. ptr node-size)))


(defmethod construct-node "Map"
  [ptr]
  (MapHandle. ptr))


(defn tvm-array
  "Called when something like a shape needs to be passed into a tvm function.  Most users will not need to call this
explicitly; it is done for you."
  [& args]
  (apply global-function "_Array" args))


(defn tvm-map
  "Call like hash-map except all keys must be node handles.  Most users will not need to call this explicitly"
  [& args]
  (when-not (= 0 (rem (count args)
                      2))
    (throw (ex-info "Map fn call must have even arg count"
                    {:args args})))
  (apply global-function "_Map" args))


(defmethod tvm-value->jvm :node-handle
  [long-val val-type-kwd]
  (-> (construct-node (Pointer. long-val))
      (resource/track #(TVMNodeFree (Pointer. long-val))
                      [:gc :stack])))


(defn get-node-type
  [node-handle]
  (get node-handle :tvm-type-kwd))


(extend-protocol bindings-proto/PJVMTypeToTVMValue
  Object
  (->tvm-value [value]
    (cond
      (sequential? value)
      (bindings-proto/->tvm-value (apply tvm-array value))
      (map? value)
      (bindings-proto/->tvm-value (apply tvm-map (->> (seq value)
                                       (apply concat))))
      (nil? value)
      [(long 0) :null])))
