(ns tvm-clj.ast
  "AST that is convertible to tvm."
  (:require [tvm-clj.base]))


(defmacro when-not-error
  [condition throw-clause]
  `(when-not (do ~condition)
     (throw ~throw-clause)))


(defn- ast-node
  [node-typename & {:keys [] :as opt-map}]
  (assoc opt-map
         :ast-type node-typename))


(defn variable
  "Create a scalar variable."
  [var-name & {:keys [dtype]
               :or {dtype "int32"}}]
  (ast-node :variable :name var-name :dtype dtype))


(defn placeholder
  "Create a tensor placeholder"
  [shape & {:keys [dtype name]
            :or {dtype "float32"
                 name "placeholder"}}]
  (ast-node :placeholder :shape shape :dtype dtype :name name))


(defn shape
  [tensor-item]
  (when-not-error (:shape tensor-item)
    (ex-info "Shape requested but object has no shape"
             {:item tensor-item})))


(def iteration-variable-types
  {
   ;; /*!
   ;; * \brief Data parallel iteration.
   ;; *  This normally corresponds to axis of Tensor.
   ;; *  Allow all IterVar manipulations.
   ;; *
   ;; * \note This does not mean the loop
   ;; *  have to be executed in parallel fashion.
   ;; */
   :data-parallel 0
   ;; /*!
   ;; * \brief The IterVar itself is a thread-index
   ;; *  of a fixed thread launching group.
   ;; *  Note that this is already assumed to be paralellized.
   ;; *
   ;; *  Disallow: split/fuse/vectorize/parallel
   ;; */
   :thread-index 1
   ;; /*!
   ;; * \brief Communicative reduction.
   ;; *  Cannot be directly parallelized.
   ;; *
   ;; *  Disallow: parallel/vectorize
   ;; */
   :communicative-reduce 2
   ;; /*!
   ;; * \brief Serial loops with loop carry dependency,
   ;; *  the iteration must execute in order.
   ;; *  Cannot be re-ordered.
   ;; *
   ;; *  Disallow: reorder/parallel/vectorize
   ;; */
   :ordered 3
   ;; /*!
   ;; * \brief IterVar is opaque,
   ;; *
   ;; *  May not corresponds to any generated loop
   ;; *  Disallow all IterVar manipulations and compute_at
   ;; *
   ;; * \note This is usually used to implement composite op
   ;; *  or external op, where the
   ;; */
   :opaque 4
   ;; // The following are possible additional
   ;; // types that are provided during schedule
   ;; /*!
   ;; * \brief The execution is unrolled.
   ;; */
   :unrolled 5
   ;; /*!
   ;; * \brief The loop is vectorized.
   ;; */
   :vectorized 6
   ;; /*!
   ;; * \brief The loop is parallelized.
   ;; */
   :parallelized 7
   ;; /*!
   ;; * \brief Marks boundary of tensorization intrinsic.
   ;; */
   :tensorized 8}
  )


(def iteration-variable-type-set (set (keys iteration-variable-types)))


(defn iteration-variable
  [domain name iteration-type & {:keys [thread-tag]
                                 :or {thread-tag ""}}]
  (when-not-error (= 2 (count domain))
    (ex-info "Domain must have 2 members"
             {:domain-count (count domain)}))

  (when-not-error (iteration-variable-type-set iteration-type)
    (ex-info "Iteration type not in allowed iteration types"
             {:allowed-types iteration-variable-type-set
              :iteration-type iteration-type}))
  (ast-node :iteration-variable :domain domain :name name :iteration-type iteration-type :thread-tag thread-tag))


(defn get-value
  [tensor indices]
  (when-not-error (= (shape tensor)
                     (count indices))
    (ex-info "Must have equal indices to the shape of tensor"
             {:tensor-shape (shape tensor)
              :indices indices}))
  (ast-node :get-value :tensor tensor :indices indices))


(defn binary-op
  [lhs rhs operation]
  (ast-node :binary-op :lhs lhs :rhs rhs :operation operation))
