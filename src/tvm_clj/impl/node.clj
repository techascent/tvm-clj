(ns tvm-clj.impl.node
  (:require [tvm-clj.impl.base :refer [make-tvm-jna-fn
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
            [tvm-clj.impl.typenames :as typenames]
            [tvm-clj.impl.protocols :as bindings-proto]
            [tech.v3.jna :refer [checknil] :as jna]
            [tech.v3.resource :as resource]
            ;;Force generation of global functions
            [tvm-clj.impl.fns.node :as node-fns]
            [tvm-clj.impl.fns.runtime :as runtime]
            [tech.v3.datatype :as dtype]
            [tech.v3.datatype.protocols :as dtype-proto]
            [clojure.tools.logging :as log])
  (:import [com.sun.jna Native NativeLibrary Pointer Function Platform]
           [com.sun.jna.ptr PointerByReference IntByReference LongByReference]
           [java.util Map List RandomAccess]
           [java.io Writer]
           [tech.v3.datatype ObjectReader ObjectBuffer]
           [clojure.lang MapEntry IFn IObj]))

(set! *warn-on-reflection* true)

(defmulti get-extended-node-value
  "Override this to enable type-specific lookups into nodes."
  (fn [node-handle item-key]
    (bindings-proto/node-type-name node-handle)))


(defmethod get-extended-node-value :default
  [& args]
  nil)


(defn get-node-fields
  [node]
  (resource/stack-resource-context
   (let [field-fn (node-fns/NodeListAttrNames node)]
     (vec (for [idx (range (field-fn -1))]
            (field-fn idx))))))


(defn get-node-field
  [node field-name]
  (node-fns/NodeGetAttr node field-name))



(make-tvm-jna-fn TVMObjectGetTypeIndex
                 "Get the type index of a node."
                 Integer
                 [node-hdl checknil]
                 [out_index int-ptr])


(defn get-node-type-index
  [^Pointer handle]
  (let [node-type-data (IntByReference.)]
    (check-call
     (TVMObjectGetTypeIndex handle node-type-data))
    (.getValue node-type-data)))


(make-tvm-jna-fn TVMObjectTypeKey2Index
                 "Convert a type name to a type index."
                 Integer
                 [type_key jna/string->ptr]
                 [out_index int-ptr])


(defn- node-type-name->index
  "Convert a node type name to an index."
  [^String type-name]
  (let [int-data (IntByReference.)]
    (check-call (TVMObjectTypeKey2Index type-name int-data))
    (.getValue int-data)))


(defonce node-type-index->name*
  (delay
    (->> typenames/typenames
         (map (fn [tname]
                (try
                  [
                   (long (node-type-name->index tname))
                   tname
                   ]
                  (catch Exception e
                    (log/warnf "Failed to find type index for name %s" tname)))))
         (remove nil?)
         (into {}))))


(declare construct-node-handle)


(deftype NodeHandle [^Pointer handle fields metadata]
  bindings-proto/PJVMTypeToTVMValue
  (->tvm-value [this]
    (let [retval [(Pointer/nativeValue handle)
                  (if (:rvalue-reference? metadata)
                    :object-rvalue-ref-arg
                    :node-handle)]]
      retval))
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
  dtype-proto/PElemwiseDatatype
  (elemwise-datatype [item] (-> (.getOrDefault item :dtype "object")
                                (keyword)))
  dtype-proto/PShape
  (shape [item] (:shape item))
  jna/PToPtr
  (is-jna-ptr-convertible? [item] true)
  (->ptr-backing-store [item] handle)
  Object
  (equals [a b]
    (if (nil? b)
      false
      (= (.hashCode a) (.hashCode b))))
  (hashCode [this]
    (-> (runtime/ObjectPtrHash this)
        Long/hashCode))
  (toString [this] (node-fns/AsRepr this))



  Map
  (containsKey [item k] (boolean (contains? fields k)))
  (entrySet [this]
    (->> (.iterator this)
         iterator-seq
         set))
  (get [this obj-key]
    (if (contains? fields obj-key)
      (get-node-field this (if (string? obj-key)
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
  IObj
  (meta [this] metadata)
  (withMeta [this newmeta]
    (NodeHandle. handle fields newmeta))
  IFn
  (invoke [this arg]
    (if (and (keyword? arg)
             (contains? fields arg))
      (.get this arg)
      (get-extended-node-value this arg)))
  (applyTo [this arglist]
    (if (= 1 (count arglist))
      (.invoke this (first arglist))
      (throw (Exception. "Too many arguments to () operator")))))


(defn is-node-handle?
  [item]
  (bindings-proto/is-node-handle? item))


(make-tvm-jna-fn TVMObjectFree
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
    (if (nil? b)
      false
      (= (.hashCode a) (.hashCode b))))
  (hashCode [this]
    (-> (runtime/ObjectPtrHash this)
        Long/hashCode))
  (toString [this] (node-fns/AsRepr this))

  ObjectReader
  (lsize [this] num-items)
  (readObject [this idx]
    (runtime/ArrayGetItem this idx)))


(defn get-map-items
  [handle]
  (->> (runtime/MapItems handle)
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
  (containsKey [item k] (not= 0 (runtime/MapCount item (bindings-proto/->node k))))
  (entrySet [this]
    (->> (.iterator this)
         iterator-seq
         set))
  (get [this obj-key]
    (let [key-node (bindings-proto/->node obj-key)]
      (when (.containsKey this key-node)
        (runtime/MapGetItem this key-node))))
  (getOrDefault [item obj-key obj-default-value]
    (if (.containsKey item obj-key)
      (.get item obj-key)
      obj-default-value))
  (isEmpty [this] (= 0 (.size this)))
  (keySet [this] (->> (map first (get-map-items this))
                      set))
  (size [this] (int (runtime/MapSize this)))
  (values [this] (map second this))
  Iterable
  (iterator [this]
    (.iterator ^Iterable (get-map-items this)))
  Object
  (equals [a b]
    (if (nil? b)
      false
      (= (.hashCode a) (.hashCode b))))
  (hashCode [this]
    (-> (runtime/ObjectPtrHash this)
        long
        (Long/hashCode)))
  (toString [this] (node-fns/AsRepr this))
  IFn
  (invoke [this arg] (.get this arg))
  (applyTo [this arglist]
    (if (= 1 (count arglist))
      (.invoke this (first arglist))
      (throw (Exception. "Too many arguments to () operator")))))


(defmethod print-method NodeHandle
  [hdl w]
  (.write ^Writer w (str hdl)))

(defmethod print-method ArrayHandle
  [hdl w]
  (.write ^Writer w (str hdl)))

(defmethod print-method MapHandle
  [hdl w]
  (.write ^Writer w (str hdl)))

(defmulti construct-node
  (fn [ptr]
    (-> (NodeHandle. ptr #{} nil)
        (bindings-proto/node-type-name))))


(defmethod construct-node :default
  [ptr]
  (NodeHandle. ptr (try (->> (NodeHandle. ptr #{} nil)
                             (get-node-fields)
                             (map keyword)
                             set)
                        (catch Exception e
                          (log/warnf e "Failed to get node fields")
                          #{}
                          ))
               nil))


(defmethod construct-node "Array"
  [ptr]
  (let [init-handle (NodeHandle. ptr #{} nil)
        node-size (long (runtime/ArraySize init-handle))]
    (ArrayHandle. ptr node-size)))


(defmethod construct-node "Map"
  [ptr]
  (MapHandle. ptr))


(defn tvm-array
  "Called when something like a shape needs to be passed into a tvm function.  Most users will not need to call this
explicitly; it is done for you."
  [& args]
  (->> (map bindings-proto/->node args)
       (apply runtime/Array)))


(defn tvm-map
  "Create tvm map of values.  Works like hash-map."
  [& args]
  (when-not (= 0 (rem (count args)
                      2))
    (throw (ex-info "Map fn call must have even arg count"
                    {:args args})))
  (->> (map bindings-proto/->node args)
       (apply runtime/Map)))


(defmethod tvm-value->jvm :node-handle
  [long-val val-type-kwd]
  (let [tptr (Pointer. long-val)
        tidx (get-node-type-index tptr)
        tname (get @node-type-index->name* tidx)]
    (condp = tname
      "runtime.String"
      (try
        (runtime/GetFFIString (NodeHandle. tptr #{} nil))
        (finally (do #_(println (format "freeing string 0x%016X" long-val))
                     (TVMObjectFree tptr))))
      "IntImm"
      (try (get-node-field (NodeHandle. tptr #{} nil) "value")
           (finally (TVMObjectFree tptr)))
      "FloatImm"
      (try (get-node-field (NodeHandle. tptr #{} nil) "value")
           (finally (TVMObjectFree tptr)))
      (-> (construct-node (Pointer. long-val))
          (resource/track {:track-type :auto
                           :dispose-fn #(do
                                          #_(println (format "Freeing object 0x%016X" long-val))
                                          (TVMObjectFree (Pointer. long-val)))})))))


(defmethod tvm-value->jvm :object-rvalue-ref-arg
  [long-val val-type-kwd]
  (let [tptr (Pointer. long-val)
        tidx (get-node-type-index tptr)
        tname (get @node-type-index->name* tidx)]
    ;;Like a regular node except we do not free these nodes
    ;;We do not free rvalue nodes coming from other places.
    (condp = tname
      "runtime.String"
      (runtime/GetFFIString (NodeHandle. tptr #{} nil))
      (let [
            ndata
            (-> (NodeHandle. (Pointer. long-val) #{} nil)
                (vary-meta assoc :rvalue-reference? true))]
        ndata))))


(extend-protocol bindings-proto/PJVMTypeToTVMValue
  Object
  (->tvm-value [value]
    (cond
      (sequential? value)
      (bindings-proto/->tvm-value (apply tvm-array value))
      (map? value)
      (bindings-proto/->tvm-value (apply tvm-map (->> (seq value)
                                                      (apply concat))))
      (fn? value)
      (let [data
            (resource/track
             (jna-base/clj-fn->tvm-fn value)
             {:track-type :stack})]
        (bindings-proto/->tvm-value data))
      (nil? value)
      [(long 0) :null])))

(defn ->dtype
  ^String [dtype-or-name]
  (cond
    (keyword? dtype-or-name)
    (name dtype-or-name)
    (string? dtype-or-name)
    dtype-or-name
    ;;punt if it is already a node
    (instance? NodeHandle dtype-or-name)
    dtype-or-name
    :else
    (throw (ex-info (format "Invalid datatype detected: %s" dtype-or-name)
                    {:dtype dtype-or-name}))))

(defonce ^:private _const-fnptr* (delay (jna-base/name->global-function "node._const")))

(defn const
  "Convert an item to a const (immediate) value"
  [numeric-value & [dtype]]
  (let [dtype (->dtype (or dtype (dtype/datatype numeric-value)))
        [long-val _ntype] (jna-base/raw-call-function @_const-fnptr* numeric-value dtype nil)]
    (construct-node (Pointer. long-val))))


(defonce str-fn* (delay (jna-base/name->global-function "runtime.String")))


(extend-protocol bindings-proto/PConvertToNode
  Boolean
  (->node [item] (const item "uint1x1"))
  Byte
  (->node [item] (const item "int8"))
  Short
  (->node [item] (const item "int16"))
  Integer
  (->node [item] (const item "int32"))
  Long
  (->node [item] (const item "int64"))
  Float
  (->node [item] (const item "float32"))
  Double
  (->node [item] (const item "float64"))
  RandomAccess
  (->node [item] (apply tvm-array (map bindings-proto/->node item)))
  Map
  (->node [item] (apply tvm-map (apply concat item)))
  String
  (->node [item]
    (let [[long-val _ntype] (jna-base/raw-call-function @str-fn* item)]
      (NodeHandle. (Pointer. (long long-val)) #{} nil)))
  Object
  (->node [item]
    (cond
      (instance? Iterable item)
      (apply tvm-array (map bindings-proto/->node item))
      :else
      (throw (Exception. (format "Object type %s is not convertible to node"
                                 (type item)))))))
