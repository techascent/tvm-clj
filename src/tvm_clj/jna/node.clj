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
                                      tvm-value->jvm]]
            [tvm-clj.bindings.definitions :refer [tvm-datatype->keyword-nothrow
                                                  node-type-name->keyword-map]]
            [tvm-clj.bindings.protocols :as bindings-proto]
            [potemkin :as p]
            [tech.jna :refer [checknil] :as jna]
            [tech.resource :as resource])
  (:import [com.sun.jna Native NativeLibrary Pointer Function Platform]
           [com.sun.jna.ptr PointerByReference IntByReference LongByReference]))



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


(defn tvm-array->jvm
  [tvm-array-node]
  (->> (range (global-function "_ArraySize" tvm-array-node))
       (mapv #(global-function "_ArrayGetItem" tvm-array-node (int %1)))))


(defn tvm-map->jvm
  [tvm-map-node]
  (->> (global-function "_MapItems" tvm-map-node)
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


(p/def-map-type NodeHandle [^Pointer tvm-jcpp-handle fields data]
  (get [this key default-value]
       (or
        (condp = key
          :tvm-jcpp-handle tvm-jcpp-handle
          :data data
          :fields fields
          (if-let [retval (get data key default-value)]
            retval
            (if (fields key)
              (let [retval (get-node-field tvm-jcpp-handle key)]
                (cond
                  (= :array (:tvm-type-kwd retval))
                  (tvm-array->jvm retval)
                  (= :map (:tvm-type-kwd retval))
                  (tvm-map->jvm retval)
                  :else
                  retval))
              (get-extended-node-value this key))))
        default-value))
  (assoc [this key value]
         (if (= key :tvm-jcpp-handle)
           (NodeHandle. value fields data)
           (NodeHandle. tvm-jcpp-handle fields (assoc data key value))))
  (dissoc [this key]
          (NodeHandle. tvm-jcpp-handle fields (dissoc data key)))
  (keys [this]
        (set (concat [:tvm-jcpp-handle] (keys data) fields)))
  (meta [this]
        (meta data))
  (with-meta [this meta]
    (NodeHandle. tvm-jcpp-handle fields (with-meta data meta)))
  ;;There is no equivalence-type system.  Two node handles are equal, equivalent
  ;;if they point to the same raw-object.  Else they aren't.  Additional data
  ;;saved on each on can't matter for the API to work correctly.
  (hasheq [this]
          (.hashCode this))
  (equiv [this other]
         (.equals this other))
  (equals [a b]
          (= (.hashCode a) (.hashCode b)))
  (hashCode [this]
            (global-function "_raw_ptr" this))
  (toString [this]
            (assoc
             (->> (keys this)
                  (map (fn [k]
                         [k (get this k)]))
                  (into {}))
             :raw-ptr (.hashCode this))))


(defn is-node-handle?
  [item]
  (instance? NodeHandle item))


(make-tvm-jna-fn TVMNodeFree
                 "Free a tvm node."
                 Integer
                 [handle checknil])


(extend-type NodeHandle
  bindings-proto/PJVMTypeToTVMValue
  (->tvm-value [item]
    [(Pointer/nativeValue (.tvm-jcpp-handle item)) :node-handle])
  bindings-proto/PToTVM
  (->tvm [item]
    item)
  bindings-proto/PConvertToNode
  (->node [item] item)
  resource/PResource
  (release-resource [item]
    (TVMNodeFree (:tvm-jcpp-handle item))))


(make-tvm-jna-fn TVMNodeGetTypeIndex
                 "Get the type index of a node."
                 Integer
                 [node-hdl checknil]
                 [out_index int-ptr])


(defn- get-node-type-index
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


(def node-type-index->keyword
  (memoize
   (fn [type-index]
     (->> node-type-name->keyword-map
          (map (fn [[type-name keyword]]
                 (when (= type-index (node-type-name->index type-name))
                   keyword)))
          (remove nil?)
          first))))


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


(defn construct-node
  ^NodeHandle [^Pointer node-handle]
  (let [fields (->> (get-node-fields node-handle)
                    (map keyword)
                    set)
        type-index (get-node-type-index node-handle)
        type-kwd (node-type-index->keyword type-index)]
    (NodeHandle. node-handle fields {:tvm-type-index type-index
                                     :tvm-type-kwd type-kwd})))


(defmethod tvm-value->jvm :node-handle
  [long-val val-type-kwd]
  (-> (construct-node (Pointer. long-val))
      resource/track))


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
