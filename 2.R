install.packages("psych")
install.packages("class")
install.packages("gmodels")
install.packages("readr")
install.packages("ggplot2")

library(psych)
library(class)
library(gmodels)
library(readr)
library(ggplot2)


#read the file
adult<-read_csv("adult.data",na="?",col_names=FALSE)
length(adult)
names(adult)
names(adult)<-c("age","workclass","fnlwgt","education","education_num","marital_status","occupation","relationship","race","sex","capital_gain","capital_loss","hours_per_week","native_country","income")


#data visualation
adult[,1:10]
adult[1:5,1:10]

barplot(table(adult$income),main='Income Classification',col='blue',ylab='No. of people')

ggplot(adult,aes(x=sex,y=income,fill=income))+geom_bar(stat = 'identity')+theme(axis.text.x = element_text(angle = 45, hjust = 1))+labs(x="gender",y="Count",title = "Income w.r.t gender")
ggplot(adult,aes(x=occupation,y=income,fill=income))+geom_bar(stat='identity')+theme(axis.text.x=element_text(angle=45,hjust=1))+labs(x="occupation",y="Count",title="Income w.r.t occupation")
ggplot(adult,aes(x=workclass,y=income,fill=income))+geom_bar(stat='identity')+theme(axis.text.x=element_text(angle=45,hjust=1))+labs(x="workclass",y="Count",title="Income w.r.t workclass")
ggplot(adult,aes(x=education,y=income,fill=income))+geom_bar(stat='identity')+theme(axis.text.x=element_text(angle=45,hjust=1))+labs(x="education",y="Count",title="Income w.r.t education")

boxplot (hours_per_week ~ income, data=adult, main="Hours Per Week distribution for different income levels",xlab="Income Levels",ylab="Hours Per Week",col="salmon")


#prepare the data
#remove.packages("ggplot2")
install.packages("factoextra")
library(factoextra)

#convert the non-numerical attributes to numbers
adult.na<-na.omit(adult)
adult.na$workclass<-as.factor(adult.na$workclass)
levels(adult.na$workclass)<-1:length(levels(adult.na$workclass))
adult.na$workclass<-as.numeric(adult.na$workclass)

adult.na$education<-as.factor(adult.na$education)
levels(adult.na$education)<-1:length(levels(adult.na$education))
adult.na$education<-as.numeric(adult.na$education)

adult.na$marital_status<-as.factor(adult.na$marital_status)
levels(adult.na$marital_status)<-1:length(levels(adult.na$marital_status))
adult.na$marital_status<-as.numeric(adult.na$marital_status)

adult.na$occupation<-as.factor(adult.na$occupation)
levels(adult.na$occupation)<-1:length(levels(adult.na$occupation))
adult.na$occupation<-as.numeric(adult.na$occupation)

adult.na$relationship<-as.factor(adult.na$relationship)
levels(adult.na$relationship)<-1:length(levels(adult.na$relationship))
adult.na$relationship<-as.numeric(adult.na$relationship)

adult.na$race<-as.factor(adult.na$race)
levels(adult.na$race)<-1:length(levels(adult.na$race))
adult.na$race<-as.numeric(adult.na$race)

adult.na$sex<-as.factor(adult.na$sex)
levels(adult.na$sex)<-1:length(levels(adult.na$sex))
adult.na$sex<-as.numeric(adult.na$sex)

adult.na$native_country<-as.factor(adult.na$native_country)
levels(adult.na$native_country)<-1:length(levels(adult.na$native_country))
adult.na$native_country<-as.numeric(adult.na$native_country)

adult.na$income<-as.factor(adult.na$income)
levels(adult.na$income)<-1:length(levels(adult.na$income))
adult.na$income<-as.numeric(adult.na$income)

#nomorlisation
normalize<-function(x){((x-min(x))/(max(x)-min(x)))}
adult.normal<-as.data.frame(lapply(adult.na[1:14],normalize))
adult.normal<-cbind(adult.normal,income=adult.na$income)

#plot(adult.normal[,1:15],lower.panel=NULL)
plot(adult.normal[,1:6],lower.panel=NULL)


#clustering
#KMeans
adult.normal.k2<-kmeans(adult.normal,centers=2,nstart=25)
factoextra::fviz_cluster(adult.normal.k2,adult.normal)

adult.normal.k3<-kmeans(adult.normal,centers=3,nstart=25)
factoextra::fviz_cluster(adult.normal.k3,adult.normal)

adult.normal.k4<-kmeans(adult.normal,centers=4,nstart=25)
factoextra::fviz_cluster(adult.normal.k4,adult.normal)

adult.normal.k5<-kmeans(adult.normal,centers=5,nstart=25)
factoextra::fviz_cluster(adult.normal.k5,adult.normal)

adult.normal.k6<-kmeans(adult.normal,centers=6,nstart=25)
factoextra::fviz_cluster(adult.normal.k6,adult.normal)

adult.normal.k7<-kmeans(adult.normal,centers=7,nstart=25)
factoextra::fviz_cluster(adult.normal.k7,adult.normal)

adult.normal.k8<-kmeans(adult.normal,centers=8,nstart=25)
factoextra::fviz_cluster(adult.normal.k8,adult.normal)

adult.normal.k9<-kmeans(adult.normal,centers=9,nstart=25)
factoextra::fviz_cluster(adult.normal.k9,adult.normal)

adult.normal.k10<-kmeans(adult.normal,centers=10,nstart=25)
factoextra::fviz_cluster(adult.normal.k10,adult.normal)

#analyze the data for clusters from k=2 to 10
factoextra::fviz_nbclust(adult.normal, FUNcluster=kmeans,print.summary=TRUE)

#kNN
#analyze the data for clusters from k=2 to 10
install.packages("kknn")
install.packages("sampling")

library(kknn)
library(sampling)

foo<-function(data=adult.norm,feats,nn=8,low=2,high=10){
  adult.n<-data[,feats]
  adult.norm.nrows<-nrow(adult.n)
  adult.norm.sample<-0.7
  adult.norm.train.index<-sample(adult.norm.nrows,adult.norm.sample*adult.norm.nrows)
  adult.norm.train<-adult.n[adult.norm.train.index,]
  adult.norm.test<-adult.n[-adult.norm.train.index,]
  
  for(nc in low:high){
    print("###############################")
    print("")
    print(nc)
    print("###############################")
    adult.norm.train.k4<-kmeans(adult.norm.train,centers=nc)
    adult.norm.train.labels<-adult.norm.train.k4$cluster
    
    adult.norm.test.k4<-kmeans(adult.norm.test,centers=nc)
    adult.norm.test.labels<-adult.norm.test.k4$cluster
    
    adult.norm.test.pred<-knn(adult.norm.train,adult.norm.test,adult.norm.train.k4$cluster,k=nn)
    str(adult.norm.test.pred)
    adult.norm.ct<-CrossTable(adult.norm.test.labels, adult.norm.test.pred,prop.chisq=FALSE)
    ##confusionMatrix(adult.norm.test.pred,adult.norm.test.labels)
  }
}
foo(adult.normal,c("age","marital_status","hours_per_week","income","education","occupation"),nn=10,2,10)

#iClust
#for(i in 2:10){
#  adult.normal.iCluster<-iclust(adult.normal,nclusters=i)
#}
adult.normal.iCluster.k2<-iclust(adult.normal,nclusters=2)
adult.normal.iCluster.k3<-iclust(adult.normal,nclusters=3)
adult.normal.iCluster.k4<-iclust(adult.normal,nclusters=4)
adult.normal.iCluster.k5<-iclust(adult.normal,nclusters=5)
adult.normal.iCluster.k6<-iclust(adult.normal,nclusters=6)
adult.normal.iCluster.k7<-iclust(adult.normal,nclusters=7)
adult.normal.iCluster.k8<-iclust(adult.normal,nclusters=8)
adult.normal.iCluster.k9<-iclust(adult.normal,nclusters=9)
adult.normal.iCluster.k10<-iclust(adult.normal,nclusters=10)


#prediction
#50-50
adult.norm.nrows<-nrow(adult.normal)
adult.norm.sample<-0.5
adult.norm.train.index<-sample(adult.norm.nrows, adult.norm.sample*adult.norm.nrows)
adult.norm.train<-adult.normal[adult.norm.train.index,]
adult.norm.test<-adult.normal[-adult.norm.train.index,]

adult.norm.train.k4<-kmeans(adult.norm.train,centers=4)
adult.norm.test.k4<-knn(adult.norm.train, adult.norm.test,adult.norm.train.k4$cluster,k=4)
adult.norm.train.tables<-adult.norm.train.k4$cluster

adult.norm.test.k4<-kmeans(adult.norm.test,centers=6)
adult.norm.test.labels<-adult.norm.test.k4$cluster
adult.norm.test.pred<-knn(adult.norm.train,adult.norm.test,adult.norm.train.k4$cluster,k=10)
str(adult.norm.test.pred)
adult.norm.ct<-CrossTable(adult.norm.test.labels,adult.norm.test.pred,prop.chisq=FALSE)

adult.norm.train.lm<-lm(formula = adult.norm.train$income~adult.norm.train$fnlwgt+adult.norm.train$education+
                          adult.norm.train$education_num+adult.norm.train$marital_status+adult.norm.train$occupation+
                          adult.norm.train$relationship+adult.norm.train$race+adult.norm.train$sex+adult.norm.train$capital_gain+
                          adult.norm.train$capital_loss+adult.norm.train$hours_per_week+adult.norm.train$native_country, 
                          data = adult.norm.train[1:14])
adult.na.lm.pred<-predict.lm(adult.norm.train.lm)
summary(adult.na.lm.pred)

adult.norm.train.glm<-glm(formula = adult.norm.train$income~adult.norm.train$fnlwgt+adult.norm.train$education+
                          adult.norm.train$education_num+adult.norm.train$marital_status+adult.norm.train$occupation+
                          adult.norm.train$relationship+adult.norm.train$race+adult.norm.train$sex+adult.norm.train$capital_gain+
                          adult.norm.train$capital_loss+adult.norm.train$hours_per_week+adult.norm.train$native_country, 
                        data = adult.norm.train[1:14])
adult.na.glm.pred<-predict.lm(adult.norm.train.glm)
summary(adult.na.glm.pred)

adult.norm.train.lm<-lm(formula = adult.norm.train$income~adult.norm.train$education+
                            adult.norm.train$education_num+adult.norm.train$marital_status+adult.norm.train$occupation+
                            adult.norm.train$relationship+adult.norm.train$race+adult.norm.train$sex+adult.norm.train$capital_gain+
                            adult.norm.train$capital_loss+adult.norm.train$hours_per_week, 
                          data = adult.norm.train[1:14])
adult.na.lm.pred<-predict.lm(adult.norm.train.lm)
summary(adult.na.lm.pred)

adult.norm.train.lm<-lm(formula = adult.norm.train$income~adult.norm.train$education_num+
                            +adult.norm.train$capital_gain+
                            adult.norm.train$capital_loss+adult.norm.train$hours_per_week, 
                          data = adult.norm.train[1:14])
adult.na.lm.pred<-predict.lm(adult.norm.train.lm)
summary(adult.na.lm.pred)

adult.norm.train.glm<-glm(formula = adult.norm.train$income~adult.norm.train$education_num+
                            +adult.norm.train$capital_gain+adult.norm.train$marital_status+adult.norm.train$sex
                            +adult.norm.train$hours_per_week+adult.norm.train$sex+adult.norm.train$relationship, 
                          data = adult.norm.train[1:14])
adult.na.glm.pred<-predict.lm(adult.norm.train.glm)
summary(adult.na.glm.pred)

#60-40
adult.norm.nrows<-nrow(adult.normal)
adult.norm.sample<-0.6
adult.norm.train.index<-sample(adult.norm.nrows, adult.norm.sample*adult.norm.nrows)
adult.norm.train<-adult.normal[adult.norm.train.index,]
adult.norm.test<-adult.normal[-adult.norm.train.index,]

adult.norm.train.k4<-kmeans(adult.norm.train,centers=4)
adult.norm.test.k4<-knn(adult.norm.train, adult.norm.test,adult.norm.train.k4$cluster,k=4)
adult.norm.train.tables<-adult.norm.train.k4$cluster

adult.norm.test.k4<-kmeans(adult.norm.test,centers=4)
adult.norm.test.labels<-adult.norm.test.k4$cluster
adult.norm.test.pred<-knn(adult.norm.train,adult.norm.test,adult.norm.train.k4$cluster,k=10)
str(adult.norm.test.pred)
adult.norm.ct<-CrossTable(adult.norm.test.labels,adult.norm.test.pred,prop.chisq=FALSE)

adult.norm.train.glm<-glm(formula = adult.norm.train$income~adult.norm.train$education_num+
                            +adult.norm.train$capital_gain+adult.norm.train$marital_status+adult.norm.train$sex
                          +adult.norm.train$hours_per_week+adult.norm.train$sex+adult.norm.train$relationship, 
                          data = adult.norm.train[1:14])
adult.na.glm.pred<-predict.glm(adult.norm.train.glm)
summary(adult.na.glm.pred)

#70-30
adult.norm.nrows<-nrow(adult.normal)
adult.norm.sample<-0.7
adult.norm.train.index<-sample(adult.norm.nrows, adult.norm.sample*adult.norm.nrows)
adult.norm.train<-adult.normal[adult.norm.train.index,]
adult.norm.test<-adult.normal[-adult.norm.train.index,]

adult.norm.train.k4<-kmeans(adult.norm.train,centers=4)
adult.norm.test.k4<-knn(adult.norm.train, adult.norm.test,adult.norm.train.k4$cluster,k=4)
adult.norm.train.tables<-adult.norm.train.k4$cluster

adult.norm.test.k4<-kmeans(adult.norm.test,centers=4)
adult.norm.test.labels<-adult.norm.test.k4$cluster
adult.norm.test.pred<-knn(adult.norm.train,adult.norm.test,adult.norm.train.k4$cluster,k=10)
str(adult.norm.test.pred)
adult.norm.ct<-CrossTable(adult.norm.test.labels,adult.norm.test.pred,prop.chisq=FALSE)

adult.norm.train.glm<-glm(formula = adult.norm.train$income~adult.norm.train$education_num+
                            +adult.norm.train$capital_gain+adult.norm.train$marital_status+adult.norm.train$sex
                          +adult.norm.train$hours_per_week+adult.norm.train$sex+adult.norm.train$relationship, 
                          data = adult.norm.train[1:14])
adult.na.glm.pred<-predict.glm(adult.norm.train.glm)
summary(adult.na.glm.pred)
