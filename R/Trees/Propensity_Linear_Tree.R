require("R6")
require("glmnet")
require("Rfast")

Node <- R6Class("Node",
                public = list(
                  ###########################
                  # self variables
                  ###########################
                  criterion = NULL,
                  min_node_size = NULL,
                  left = NULL,
                  right = NULL,
                  label = NULL,
                  info_gain = NULL,
                  num_samples = NULL,
                  num_classes = NULL,
                  terminal = NULL,
                  depth = NULL,
                  ratio_of_minority = NULL,
                  prop_score = NULL, #propensity score.
                  ce = NULL, # causal effect in the leaf
                  ce_lm = NULL, # causal effect in the leaf modified by linear regression
                  feature = NULL, #feature that splitting
                  threshold = NULL, # threshold of feature
                  
                  ############################
                  #  functions
                  ############################
                  
                  
                  #INITIALIZE
                  initialize = function(criterion,min_node_size,ratio_of_minority){
                    self$criterion = criterion
                    self$min_node_size = min_node_size
                    self$ratio_of_minority = ratio_of_minority
                    self$terminal = FALSE
                  },
                  
                  # ------------ START SPLIT NODE -------------#
                  # input: 
                  # x : covariate
                  # a : treatment
                  # y : outcome
                  split_node = function(x,a,y,depth,label){
                    
                    self$depth = depth 
                    self$label = label
                    
                    #サンプルサイズ
                    self$num_samples = length(a)
                    
                    #ノード内の各クラスサンプル数を計算する
                    self$num_classes = c(sum(a==0),sum(a==1))
                    
                    #ノード内のPropensity Scoreを計算する
                    self$prop_score = self$num_classes[2]/self$num_samples
                    
                    #純度が1の場合はここで関数を終了する
                    if(length(unique(a))==1){ 
                      self$terminal = TRUE
                      # causal effect 
                      self$ce = mean(y[a==1]) - mean(y[a==0])
                      
                      # linear modified causal effect
                      if(dim(x)[[2]] < 5){
                        #xの次元が大きくないならそのまま線形回帰
                        self$ce_lm <- lm(y~a+x)$coef[2]
                      }else{
                        #xの次元が大きい場合はLasso/AIC -> linear regression
                        self$ce_lm <- self$selectBIC(x,a,y) #self$selectLASSO(x,a,y)
                      }
                      return(self)
                    }
                    
                    #純度が1ではない場合は，以下で新たなsplit実行する
                    tmp <- table(a)
                    self$info_gain = -100
                    
                    # xの各列についてsplitを実行する
                    
                    # 変数の数が多い場合はすべての変数を探索するのは時間がかかるので、
                    # logistic regression modelを当てはめて90%有意な変数のみを候補として用いる
                    #if(dim(x)[[2]]>5){
                      #cfn : candidate feature index
                    #  cfi <- which(abs(summary(glm(a ~ x,family="binomial"))$coefficients[2:(dim(x)[[2]]+1),3])>qnorm(0.95))
                    #}else{
                    #  cfi <- c(1:dim(x)[[2]])
                    #}
                    cfi <- c(1:dim(x)[[2]])
                    
                    for(f in cfi){
                      uniq_feature = sort_unique(x[,f])
                      split_points = (uniq_feature[-1]+uniq_feature[-length(uniq_feature)])/2
                      if(length(split_points) > 100){
                        split_points <- split_points[seq(1,length(split_points),floor(length(split_points)/100))]
                      }
                      for(threshold in split_points){
                        l_index <- x[,f] <= threshold # left node index
                        r_index <- x[,f] > threshold #right node index
                        
                        #left node data
                        y_l = y[l_index];a_l = a[l_index];x_l = x[l_index,]
                        
                        #right node data
                        y_r = y[r_index];a_r = a[r_index];x_r = x[r_index,]
                        
                        #最小ノードサイズを超えているか
                        condition1 <- sum(a_l) > self$min_node_size & sum(1-a_l) > self$min_node_size &
                          sum(a_r) > self$min_node_size & sum(1-a_r) > self$min_node_size
                        
                        #ノード内のマイノリティーサンプルはノード内のサンプルに対する比の最小ラインを超えるか
                        # 具体的には
                        # 1) 親ノードのないのW=1のサンプルのうちの100α%以上のW=1サンプルが子ノードに含まれるか
                        # 2) 親ノードのないのW=0のサンプルのうちの100α%以上のW=0サンプルが子ノードに含まれるか
                        # の2つを左右のノードでチェックする。
                        condition_for_minority_ratio <- 
                          sum(a_l)/sum(a) > self$ratio_of_minority & # Left node の W=1
                          sum(1-a_l)/sum(1-a) > self$ratio_of_minority & # Left node の W=0
                          sum(a_r)/sum(a) > self$ratio_of_minority & # Right node の W=1
                          sum(1-a_r)/sum(1-a) > self$ratio_of_minority # Right node の W=0
                        
                        if(condition1 & condition_for_minority_ratio){
                          val = self$split_loss(x,x_l,x_r,a,a_l,a_r,y,y_l,y_r)
                          if(self$info_gain < val){
                            self$info_gain = val
                            self$threshold = threshold
                            self$feature = f
                          }else{
                            self$info_gain = self$info_gain
                            self$threshold = self$threshold
                            self$feature = self$feature
                          } # end if-else
                        } #end if condition
                      }# end threshold for
                    }# end cfi for
                    
                    
                    
                    # 結果の分割を保存
                    a_l = a[x[,self$feature] <= self$threshold]
                    a_r = a[x[,self$feature] > self$threshold]
                    
                    condition2 <- sum(a_l) < self$min_node_size | 
                      sum(1-a_l) < self$min_node_size |
                      sum(a_r) < self$min_node_size | 
                      sum(1-a_r) < self$min_node_size
                    
                    if(self$info_gain==-100){
                      self$terminal = TRUE
                      # causal effect 
                      self$ce = mean(y[a==1]) - mean(y[a==0])
                      
                      # linear modified causal effect
                      if(dim(x)[[2]] < 5){
                        #xの次元が大きくないならそのまま線形回帰
                        self$ce_lm <- lm(y~a+x)$coef[2]
                      }else{
                        #xの次元が大きい場合はLasso/AIC -> linear regression
                        self$ce_lm <- self$selectBIC(x,a,y) #self$selectLASSO(x,a,y)
                      }
                      return(self)
                    }
                    
                    if(condition2){
                      self$terminal = TRUE
                      # causal effect 
                      self$ce = mean(y[a==1]) - mean(y[a==0])
                      
                      # linear modified causal effect
                      if(dim(x)[[2]] < 5){
                        #xの次元が大きくないならそのまま線形回帰
                        self$ce_lm <- lm(y~a+x)$coef[2]
                      }else{
                        #xの次元が大きい場合はLasso/AIC -> linear regression
                        self$ce_lm <- self$selectBIC(x,a,y) #self$selectLASSO(x,a,y)
                      }
                      return(self)
                    }
                    #再帰分割(recursive partition)を実行する
                    #Left node:
                    x_l = x[x[,self$feature] <= self$threshold,]
                    a_l = a[x[,self$feature] <= self$threshold]
                    y_l = y[x[,self$feature] <= self$threshold]
                    self$left = Node$new(self$criterion,self$min_node_size,self$ratio_of_minority)
                    self$left$split_node(x_l,a_l,y_l,depth=depth+1,label=2*label)
                    
                    
                    #Right node:
                    x_r = x[x[,self$feature] > self$threshold,]
                    a_r = a[x[,self$feature] > self$threshold]
                    y_r = y[x[,self$feature] > self$threshold]
                    self$right = Node$new(self$criterion,self$min_node_size,self$ratio_of_minority)
                    self$right$split_node(x_r,a_r,y_r,depth=depth+1,label=2*label+1)
                  }, 
                  # ---------- END SPLIT NODE FUNCTION ----------- #
                  
                  # -------- Splitting loss function (likelihood based)---------- #
                  
                  split_loss = function(x,x_l,x_r,a,a_l,a_r,y,y_l,y_r){

                    # number of samples in the node.
                    N_p = length(a);N_p1 = sum(a==1);
                    N_cl = length(a_l);N_cl1 = sum(a_l==1);
                    N_cr = length(a_r);N_cr1 = sum(a_r==1);
                    
                    loglik_parent = sum(a*log(N_p1/N_p)+(1-a)*log(1-N_p1/N_p))
                    loglik_left  = sum(a_l*log(N_cl1/N_cl)+(1-a_l)*log(1-N_cl1/N_cl))
                    loglik_right = sum(a_r*log(N_cr1/N_cr)+(1-a_r)*log(1-N_cr1/N_cr))
                    
                    val = loglik_left + loglik_right - loglik_parent
                    return(val)
                  },
                  
                  #regression model selection step
                  selectBIC = function(x,a,y){
                    x = as.matrix(x)
                    colnames(x) <- paste0("x",c(1:dim(x)[[2]]))
                    dd <- data.frame(x,a,y)
                    lm_r <- lm(y~a,data=dd)
                    u_formula <- formula(paste0("~a+",stringr::str_c(paste0("x",c(1:dim(x)[[2]])),collapse ="+")))
                    res <- step(lm_r,scope=list(upper=u_formula, lower=formula("~a")),direction="forward",k=log(length(y)),trace=FALSE)
                    return(res$coef[2])
                  },
                  
                  selectLASSO = function(x,a,y){
                    dim_p = dim(x)[[2]]
                    lasso.model.cv <- cv.glmnet(x = as.matrix(cbind(a,x)), y = y, family = "gaussian", alpha = 1,nfold=5)
                    lasso.model <- glmnet(x = as.matrix(cbind(a,x)), y = y, family = "gaussian",lambda = lasso.model.cv$lambda.min, alpha = 1)
                    #有効な変数がない場合はただの線形回帰
                    if(sum(lasso.model$beta[-1] > 0)==0){
                      return(lm(y~a)$coef[2])
                    }
                    #有効な変数がある場合はただの線形回帰
                    if(sum(lasso.model$beta[-1] > 0)!=0){
                      x_eff <- x[,which(lasso.model$beta[-1] > 0)]
                      return(lm(y~a+x_eff)$coef[2])
                    }
                  }, 
                  
                  # split_loss = function(x,x_l,x_r,a,a_l,a_r,y,y_l,y_r){
                  # 
                  #   # number of samples in the node.
                  #   N_p = length(a);N_p1 = sum(a==1);
                  #   N_cl = length(a_l);N_cl1 = sum(a_l==1);
                  #   N_cr = length(a_r);N_cr1 = sum(a_r==1);
                  # 
                  #   resid_p <- lm(y~a+x)$resid
                  #   resid_l <- lm(y_l~a_l+x_l)$resid
                  #   resid_r <- lm(y_r~a_r+x_r)$resid
                  # 
                  #   cri_p = var(resid_p)
                  #   cri_l = var(resid_l)
                  #   cri_r = var(resid_r)
                  # 
                  #   if(is.na(cri_p)){
                  #     print(c(length(y),length(a)))
                  #   }
                  #   val = cri_p - length(y_l)/length(y)*cri_l - length(y_r)/length(y)*cri_r
                  #   return(val)
                  # },
                  
                  
                  # -------- START PREDICT FUNCTION ---------#
                  
                  predict = function(x){
                    if(self$terminal){
                      return(c(self$ce,self$ce_lm))
                    }else{
                      if(x[self$feature] <= self$threshold){
                        return(self$left$predict(x))
                      }else{
                        return(self$right$predict(x))
                      }
                    }
                  }
                  
                ) #END PUBLIC LIST
) #END NODE CLASS


MIG_subclass <- R6Class("MIG",
                        public = list(
                          ###########################
                          # self variables
                          ###########################
                          tree = NULL,
                          criterion = NULL,
                          min_node_size = NULL,
                          ratio_of_minority = NULL, #n_1k / n_1k + n_0k, #n_0k / n_1k + n_0kが超えてほしい値
                          
                          ############################
                          #  functions
                          ############################
                          initialize = function(criterion="gini",min_node_size = NULL, ratio_of_minority = 0){
                            self$tree = NULL
                            self$criterion = criterion
                            self$min_node_size = min_node_size
                            self$ratio_of_minority = ratio_of_minority
                          },
                          
                          #当てはめ結果を返す関数
                          fit = function(x,a,y){
                            self$tree = Node$new(self$criterion,self$min_node_size,self$ratio_of_minority)
                            self$tree$split_node(x,a,y,depth=1,label=1)
                          },
                          
                          #予測結果を返す関数
                          predict = function(x){
                            iter = dim(x)[[1]]
                            pred = matrix(0,iter,2) #causal effect + causal effect with linear modification
                            for(s in 1:iter){
                              pred[s,] = self$tree$predict(x[s,])
                            }
                            return(pred)
                          }

                        )
)