# Classification And Regression Trees (CART)
# This code is for classification that proposed by breiman (1984)
# criterion function : "Gini index" or "entropy" can be selected
# x : p-dim covariates 
# y : outcome
# usage :
# x <- iris[,-5]
# y <- iris$Species
# clf_m <- DecisionTree(criterion="gini",min_node_size = 5, ratio_of_minority = 0.1)
# clf_m$fit(x,y)
# result <- clf_predict(x)

# @param
# @
# @
# @
# @

require("R6")
require("Rfast")

Node <- R6Class("Node",
  public = list(
    ###########################
    # self variables
    ###########################
    criterion = NULL,
    max_depth = NULL,
    min_node_size = NULL,
    alpha_regular = NULL,
    mtry = NULL,
    random_state = NULL,
    depth = NULL,
    left = NULL,
    right = NULL,
    feature = NULL,
    threshold = NULL,
    label = NULL,
    impurity = NULL,
    info_gain = NULL,
    num_samples = NULL,
    num_classes = NULL,
    terminal = NULL,
    
    ############################
    #  functions
    ############################
    
    
    #INITIALIZE
    initialize = function(criterion,max_depth,min_node_size,alpha_regular,mtry,random_state){
      self$criterion = criterion
      self$max_depth = max_depth
      self$min_node_size = min_node_size
      self$alpha_regular = alpha_regular
      self$mtry = mtry
      self$random_state = random_state
      #depthとmin_node_sizeが指定された場合に以下の警告を実行
      print("parameter depth and min_node_size are both detected. Prioritize DEPTH parameter")
    },
    
    # ------------ START SPLIT NODE -------------#
    split_node = function(sample,target,depth,unique_class){
      #木の深さ、サンプル数、クラスに属するサンプル数を計算
      self$depth = depth
      self$num_samples = length(target)
      self$num_classes = rep(0,length(unique_class))
      for(j in 1:length(unique_class)){
        self$num_classes[j] = length(target[target == unique_class[j]])
      }
      
      #Gini index or Entropyが0の場合には、ここで関数を終了する
      if(length(unique(target))==1){ 
        self$label = names(which.max(table(target)))
        self$impurity = self$criterion_func(target)
        self$terminal = TRUE
        print(self$depth)
        print(self$label)
        return(self)
      }
      
      #depth < max_depthなら関数を終了する
      if(depth > self$max_depth){ 
        self$label = names(which.max(table(target)))
        self$impurity = self$criterion_func(target)
        self$terminal = TRUE
        print(self$depth)
        print(self$label)
        return(self)
      }
      
      ##############################################################
      #splittingの終了条件が満たされなかったのでsplittingを実行する#
      ##############################################################
      self$terminal = FALSE
      num_features = dim(sample)[2]
      self$info_gain = 0.0
      
      #切る変数の数を決定する(mtry)
      if(is.null(self$random_state)!=TRUE){
        set.seed(self$random_state)
      }
      if(is.null(self$mtry)==TRUE){
        mtry = num_features
      }else{
        mtry = min(1+rpois(self$mtry),num_features)
      }
      #mtryで選択された数だけ{1,2,...,p}からランダムにサンプルをとる。
      f_loop_order = sample(c(1:num_features),mtry,replace=FALSE)
      for(f in f_loop_order){
        uniq_feature = unique(sample[,f])
        split_points = (uniq_feature[-1]+uniq_feature[-length(uniq_feature)])/2
        #maximize impurity
        for(threshold in split_points){
          target_l = target[sample[,f] <= threshold]
          target_r = target[sample[,f] > threshold]
          val = self$calc_info_gain(target,target_l,target_r)
          if(self$info_gain < val){
            self$info_gain = val
            self$feature = f
            self$threshold =threshold
          }
        }
      }
      #どの変数で切っているか知りたければこれを実行．
      print(paste("feature:",self$feature,"at threshold:",self$threshold))
      
      #再帰分割(recursive partition)を実行する
        #Left node:
        sample_l = sample[sample[,self$feature]<=self$threshold,]
        target_l = target[sample[,self$feature]<=self$threshold]
        self$left = Node$new(self$criterion,
                             self$max_depth,
                             self$min_node_size,
                             self$alpha_regular,
                             self$mtry,
                             self$random_state)
        self$left$split_node(sample_l,target_l,depth+1,unique_class)
        
        
        #Right node:
        sample_r = sample[sample[,self$feature] > self$threshold,]
        target_r = target[sample[,self$feature] > self$threshold]
        self$right = Node$new(self$criterion,
                              self$max_depth,
                              self$min_node_size,
                              self$alpha_regular,
                              self$mtry,
                              self$random_state)
        self$right$split_node(sample_r,target_r,depth+1,unique_class)
    },
    # ---------- END SPLIT NODE FUNCTION ----------- #
    
    #ノード内の純度を計算する関数
    
    # ---------- START CRITERION FUNC ------------#
    criterion_func = function(target){
      classes = unique(target)
      num_data = length(target)
      
      if(self$criterion == "gini"){
        val = 1
        for(c in classes){
          p = length(target[target==c])/num_data
          val = val - p**2
        }
      } #END IF GINI
      
      if(self$criterion == "entropy"){
        val = 0
        for(c in classes){
          p = length(target[target==c])/num_data
          if(p != 0){
            val = val - p*log(p)
          }
        }
      } #END IF ENTROPY
      
      return(val)
    }, 
    # --------- END CRITERION FUNCTION ----------#
    
    
    #情報利得を計算する関数
    
    # -------- START CALC INFORMATION GAIN ---------- #
    
    calc_info_gain = function(target_p,target_cl,target_cr){
      cri_p = self$criterion_func(target_p)
      cri_cl = self$criterion_func(target_cl)
      cri_cr = self$criterion_func(target_cr)
      val = cri_p - length(target_cl)/length(target_p)*cri_cl - length(target_cr)/length(target_p)*cri_cr
      return(val)
    },
    # -------- END INFORMATION GAIN FUNCTION --------- # 
    
    #各サンプル（注意：各サンプル1つ1つ）に対して予測を返す関数
    #入力は行列ではなくてベクトルです．
    
    # -------- START PREDICT FUNCTION ---------#
    
    predict = function(sample){
      if(self$terminal){
        return(self$label)
      }else{
        if(sample[self$feature] <= self$threshold){
          return(self$left$predict(sample))
        }else{
          return(self$right$predict(sample))
        }
      }
    }
  ) #END PUBLIC LIST
) #END NODE CLASS

# --------------START CLASS TREE ANALYSIS ------------------ #
TreeAnalysis = R6Class("TreeAnalysis",
    public = list(
      ###########################
      # self variables
      ###########################
      num_features = NULL,
      importances = NULL,
      
      ############################
      #  functions
      ############################
      
      #INITIALIZE
      initialize = function(num_features,importances){
        num_features = NULL
        importances = NULL
      },
      
      #feature importance の計算関数
      compute_feature_importances = function(node){
        if(is.null(node$feature)){
          return(self)
        }
        self$importances[node$feature] =  self$importances[node$feature] + node$info_gain*node$num_samples
        self$compute_feature_importances(node$left)
        self$compute_feature_importances(node$right)
        
      },
      
      #feature importance を全体で計算する関数
      get_feature_importances = function(node,num_features,normalize=TRUE){
        self$num_features = num_features
        self$importances = rep(0,num_features)
        
        self$compute_feature_importances(node)
        self$importances = self$importances/node$num_samples
        
        if(normalize){
          normalizer = sum(self$importances)
          if(normalizer >0){
            self$importances = self$importances/normalizer
          }
        }
        return(self$importances)
      }
      
    ) #END PUBLIC
                      
                      
) #END TREE ANALYSIS CLASS



DecisionTree <- R6Class("DecisionTree",
  public = list(
    ###########################
    # self variables
    ###########################
    tree = NULL,
    criterion = NULL, #criterion function that Gini index or entropy
    max_depth = NULL, #maximum depth of the tree
    min_node_size = NULL, #minimum number of samples that contain each terminal node that k < |#L| < 2k-1
    alpha_regular = NULL, #alpha-regular parameter on Wager and Athey (2018)
    mtry = NULL, #Number of covariates that uses for each split mtry ~ rpoisson(mtry)
    random_state = NULL, # if we use mtry, prefer to set this random state parameter for reproducibility 
    tree_analysis = NULL, # can not use (not-implemented)
    feature_importances_ = NULL, # feature importance for single tree using Breiman-Kulter approach,
    
    ############################
    #  functions
    ############################
    initialize = function(criterion="gini",
                          max_depth = NULL,
                          min_node_size = NULL,
                          alpha_regular = NULL,
                          mtry = NULL,
                          random_state = NULL){
      self$tree = NULL
      self$criterion = criterion
      self$max_depth = max_depth
      self$min_node_size = min_node_size
      self$alpha_regular = alpha_regular
      self$mtry = mtry
      self$random_state = random_state
      self$tree_analysis = TreeAnalysis$new()
    },
    
    #当てはめ結果を返す関数
    fit = function(sample,target){
      self$tree = Node$new(self$criterion,
                           self$max_depth,
                           self$min_node_size,
                           self$alpha_regular,
                           self$mtry,
                           self$random_state)
      self$tree$split_node(sample,target,0,unique(target))
      self$feature_importances_ = self$tree_analysis$get_feature_importances(self$tree,dim(sample)[2])
    },
    
    #予測結果を返す関数
    predict = function(sample){
      iter = dim(sample)[1]
      pred = rep(0,iter)
      for(s in 1:iter){
        pred[s] = self$tree$predict(sample[s,])
      }
      return(pred)
    },
    
    #scoreを計算する
    score = function(sample,target){
      return(sum(self$predict(sample)==target)/length(target))
    }
  )
)