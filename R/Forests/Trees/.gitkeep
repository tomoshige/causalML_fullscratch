##' ## Classification forest with default settings
##' ranger(Species ~ ., data = iris)
##'
##' ## Prediction
##' train.idx <- sample(nrow(iris), 2/3 * nrow(iris))
##' iris.train <- iris[train.idx, ]
##' iris.test <- iris[-train.idx, ]
##' rg.iris <- ranger(Species ~ ., data = iris.train)
##' pred.iris <- predict(rg.iris, data = iris.test)
##' table(iris.test$Species, pred.iris$predictions)
##' 
##' ## Quantile regression forest
##' rf <- ranger(mpg ~ ., mtcars[1:26, ], quantreg = TRUE)
##' pred <- predict(rf, mtcars[27:32, ], type = "quantiles")
##' pred$predictions
##'
##' ## Variable importance
##' rg.iris <- ranger(Species ~ ., data = iris, importance = "impurity")
##' rg.iris$variable.importance
##'
##' ## Survival forest
##' require(survival)
##' rg.veteran <- ranger(Surv(time, status) ~ ., data = veteran)
##' plot(rg.veteran$unique.death.times, rg.veteran$survival[1,])
##'
##' ## Alternative interfaces (same results)
##' ranger(dependent.variable.name = "Species", data = iris)
##' ranger(y = iris[, 5], x = iris[, -5])



#Decision Tree
require("R6")
require("Rfast")

Node <- R6Class("Node",
  public = list(
    ###########################
    # self variables
    ###########################
    criterion = NULL,
    max_depth = NULL,
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
    
    ############################
    #  functions
    ############################
    
    
    #INITIALIZE
    initialize = function(criterion,max_depth,random_state){
      self$criterion = criterion
      self$max_depth = max_depth
      self$random_state = random_state
    },
    
    # ------------ START SPLIT NODE -------------#
    split_node = function(sample,target,depth,ini_num_classes){
      self$depth = depth
      self$num_samples = length(target)
      
      #ノード内の各クラスサンプル数を計算する
      tmp = rep(0,length(ini_num_classes));j = 1
      for(i in ini_num_classes){
        tmp[j] = length(target[target == i])
        j = j+1
      }
      self$num_classes = tmp #if there are more than 2 classes change here to below
      
      
      #親ノードのクラスを決定する
      
      #純度が1の場合はここで関数を終了する
      if(length(unique(target))==1){ 
        self$label = target[1]
        self$impurity = self$criterion_func(target)
        return(self)
      } 
      
      #純度が1ではない場合は，以下で新たなsplit実行する
      tmp <- table(target)
      self$label = as.numeric(dimnames(tmp)$target[[which(tmp == max(tmp))[1]]])
      self$impurity = self$criterion_func(target)
      
      num_features = dim(sample)[2]
      self$info_gain = 0.0
      
      #切る変数の選択(random-split)
      repeat{
        f = sample(seq(1,num_features,1),1)
        uniq_feature = sort_unique(sample[,f])
        if(length(uniq_feature)!=1){break}
      }#split pointが計算できない場合は実行しない
      split_points = (uniq_feature[-1]+uniq_feature[-length(uniq_feature)])/2
      
      for(threshold in split_points){
        target_l = target[sample[,f] <= threshold]
        target_r = target[sample[,f] > threshold]
        val = self$calc_info_gain(target,target_l,target_r)
        if(self$info_gain < val){
          self$info_gain = val
          self$feature = f
          self$threshold = threshold
        }
      }
      
      #どの変数で切っているか知りたければこれを実行．
      print(paste("feature:",self$feature,"at threshold:",self$threshold))
      
      
      #切る変数の選択(rpart的な全変数探索)
      #if(!is.null(self$random_state)){
      #  set.seed(self$random_state)
      #}
      #f_loop_order = sample(c(1:dim(sample)[2]),dim(sample)[2],replace=FALSE)
      #for(f in f_loop_order){
      #  uniq_feature = np.unique(sample[,f])
      #  split_points = (uniq_feature[-1]+uniq_feature[-1])
        
        #探索
      #  for(threshold in split_points){
      #    target_l = target[sample[,f] <= threshold]
      #    target_r = target[sample[,f] > threshold]
      #    val = self$calc_info_gain(target,target_l,target_r)
      #    if(self$info_gain < val){
      #      self$info_gain = val
      #      self$feature = f
      #      self$threshold =threshold
      #   }
      #}
        
      if(self$info_gain==0.0){
        return(self)
      }
      if(depth == self$max_depth){
        return(self)
      }
      
      #再帰分割(recursive partition)を実行する
        #Left node:
        sample_l = sample[sample[,self$feature]<=self$threshold,]
        target_l = target[sample[,self$feature]<=self$threshold]
        self$left = Node$new(self$criterion,self$max_depth,random_state=random_state)
        self$left$split_node(sample_l,target_l,depth+1,ini_num_classes)
        
        
        #Right node:
        sample_r = sample[sample[,self$feature] > self$threshold,]
        target_r = target[sample[,self$feature] > self$threshold]
        self$right = Node$new(self$criterion,self$max_depth,random_state=random_state)
        self$right$split_node(sample_r,target_r,depth+1,ini_num_classes)
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
      if(is.null(self$feature) | self$depth == self$max_depth){
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
    criterion = NULL,
    max_depth = NULL,
    random_state = NULL,
    tree_analysis = NULL,
    feature_importances_ = NULL,
    
    ############################
    #  functions
    ############################
    initialize = function(criterion="gini",max_depth = NULL, random_state = NULL){
      self$tree = NULL
      self$criterion = criterion
      self$max_depth = max_depth
      self$random_state = random_state
      self$tree_analysis = TreeAnalysis$new()
    },
    
    #当てはめ結果を返す関数
    fit = function(sample,target){
      self$tree = Node$new(self$criterion,self$max_depth,self$random_state)
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
