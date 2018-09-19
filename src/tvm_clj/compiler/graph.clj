(ns tvm-clj.compiler.graph
  (:require [clojure.set :as c-set]
            [tech.compute.tensor.dimensions :refer [when-not-error]]))


(defn compute-graph
  []
  {:nodes {}
   :edges []})


(defn edges
  [graph]
  (get graph :edges))


(defn edges->map
  [graph key-fn val-fn]
  (->> (edges graph)
       (mapcat (fn [edge]
                 (for [in (:input edge)
                       out (:output edge)]
                   [in out])))
       (group-by key-fn)
       (map (fn [[k v]]
              [k (map val-fn v)]))
       (into {})))


(defn parent->child-map
  [graph]
  (edges->map graph first second))


(defn child->parent-map
  [graph]
  (edges->map graph second first))


(defn roots
  [graph]
  (let [p->c (parent->child-map graph)]
    (c-set/difference (set (keys p->c))
                      (set (apply concat (vals p->c))))))


(defn leaves
  [graph]
  (let [c->p (child->parent-map graph)]
    (c-set/difference (set (keys c->p))
                      (set (apply concat (vals c->p))))))


(defn get-node
  [graph node-id]
  (let [retval (get-in graph [:nodes node-id])]
    (when-not-error retval
      "Failed to find node:"
      {:node-id node-id
       :nodes (keys (get graph :nodes))})
    retval))


(defn get-tensor
  [graph id]
  (let [retval (get-node graph id)]
    (when-not-error (= :tensor (:type retval))
      "Node is not tensor"
      {:node-id id
       :node retval})
    retval))


(defn generate-id
  [id-stem id-set]
  (loop [idx 1]
    (let [new-id (-> (format "%s-%s" id-stem idx)
                     keyword)]
      (if (contains? id-set new-id)
        (recur (inc idx))
        new-id))))


(defn generate-derived-node-id
  "Generate a derived node id and return the node"
  [node graph]
  (let [node-id (name (:id node))
        num-idx (.lastIndexOf node-id "-")
        ^String id-stem (if (not= -1 num-idx)
                          (.substring node-id 0 num-idx)
                          node-id)
        graph-nodes (:nodes graph)]
    (loop [idx 1]
      (let [new-id (-> (format "%s-%s" id-stem idx)
                       keyword)]
        (if (contains? graph-nodes new-id)
          (recur (inc idx))
          new-id)))))


(defn get-or-create-node-id
  "Generate an id for this node."
  [graph node]
  (when-not-error (or (nil? (:id node))
                              (nil? (get-in graph [:nodes (:id node)])))
    "Node has ID and it is already in the graph"
    {:new-node node
     :existing-node (get-node graph (:id node))})
  (assoc node :id (generate-id (name (get node :type))
                               (set (keys (get graph :nodes))))))


(defn contains-id?
  [graph id]
  (get-in graph [:nodes id]))


(defn add-node
  [graph node]
  (when-not-error (not (contains-id? graph (:id node)))
    "Graph contains node."
    {:node-ids (map :id (:nodes graph))
     :node node})
  (assoc-in graph [:nodes (:id node)] node))
