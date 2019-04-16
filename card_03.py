#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
import seaborn as sns
mpl.rcParams['font.sans-serif'] = ['SimHei']
import time
import datetime
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings    
warnings.filterwarnings("ignore") 


# In[6]:


filepath = 'E:/BaiduNetdiskDownload/jzzz/train/subsidy_train.txt'
subsidy_data=pd.read_table(filepath,sep = ',',names = ['user_id', 'subsidy'],header=None)
subsidy_data.head()


# In[8]:


len(subsidy_data.user_id.unique())


# In[10]:


subsidy_data.subsidy.value_counts()


# In[13]:


subsidy_data.subsidy.value_counts().plot.pie(labels=['0','1000','1500','2000'],
                                            autopct='%.2f',fontsize=10,figsize=(6,6),subplots=True)
plt.axis('equal')
plt.legend()
plt.title('助学金领取比例图')


# In[36]:


###引入成绩数据
name=['user_id','faculty','rank']
score_train=pd.read_csv(r'E:\BaiduNetdiskDownload\jzzz\train\score_train.txt',sep=',',names=name,header=None)
score_test=pd.read_csv(r'E:\BaiduNetdiskDownload\jzzz\test\score_test.txt',sep=',',names=name,header=None)
score_total=score_train.append(score_test)


# In[37]:


# 计算rank_rate方法
total=score_total.groupby(['faculty'])['rank'].max().reset_index()
total.columns=['faculty','total_people']
#
score_total= pd.merge(score_total,total, how='inner',on='faculty').sort_values(['user_id']).reset_index(drop=True)
score_total['rank_rate']=score_total['rank']/score_total['total_people']
score_total.head()


# In[38]:


score_total['rank_per']=score_total['rank_rate']*100
score_total=score_total[['user_id','rank_per']]


# In[39]:


##  成绩分级函数
def make_label(rank_rate):
    if rank_rate <=10:
        label=11
    elif 10 < rank_rate <= 30:
        label=12
    elif 30 < rank_rate <= 60:
        label=13
    elif 60 < rank_rate <= 80:
        label=14
    else:
        label=15
    return label    


# In[40]:


score_total['label'] = score_total.rank_per.apply(make_label)


# In[46]:


score_total.to_csv('E:/BaiduNetdiskDownload/jzzz/score_total.csv',sep=',', header=True,index=False)


# In[52]:


score_subsidy = pd.merge(score_total, subsidy_data,how= 'inner',on='user_id')


# In[54]:


len(score_subsidy)


# In[53]:


score_subsidy.head()


# In[55]:


score_subsidy.boxplot(column='rank_per',by='subsidy')
## 证明领取助学金的同学成绩反而相对落后


# In[56]:


filepath='E:/BaiduNetdiskDownload/jzzz/card_total.csv'
card_total=pd.read_csv(filepath)


# In[57]:


card_total.head()


# In[58]:


len(card_total)


# In[62]:


card_total.consume.unique()
card_total.kind.unique()


# In[73]:


###计算在校天数
card_total['date']=card_total['time'].str[0:10]
card_day=card_total[['user_id','date']]
card_day = card_day.drop_duplicates().reset_index(drop=True) #去重
card_day_group=card_day.groupby(['user_id'])['date'].count().reset_index()


# In[74]:


card_day_group.head()


# In[77]:


del card_total['date']


# In[79]:


###判断学生的消费水平，消费结构以及恩格尔系数
#消费数额有正有负，需处理
card_total['amount']=card_total['amount'].abs()


# In[80]:


card_total_consume=card_total[card_total['consume']=='POS消费'].reset_index(drop=True)  # 只选取有pos消费的一部分


# In[83]:


card_total_consume.head()


# In[85]:


# card_total_consume.isnull().sum()
##kind
card_total_consume[card_total_consume.kind.isnull().values==True][:5]
##根据消费时间的逻辑，用前一个的kind替换缺失值
card_total_consume.kind.fillna(method='pad',inplace=True)


# In[125]:


#消费结构
consume_structure = card_total_consume.groupby(['user_id','kind'])['amount'].sum()


# In[126]:


consume_structure = consume_structure.unstack('kind')# 调换行列
consume_structure.fillna(0,inplace=True)    #用0来填补缺失值
consume_structure = consume_structure


# In[127]:


consume_structure['合计'] = consume_structure.sum(axis=1)  # 注意反复运行导致合计累计叠加   ？？


# In[128]:


consume_structure.head()


# In[119]:


##特征值
consume_structure.合计.describe()
consume_structure.食堂.describe()


# In[130]:


consume_structure.食堂.hist(bins=100)


# In[131]:


consume_structure.超市.hist(bins=100)


# In[132]:


consume_structure.合计.hist(bins=100)


# In[133]:


###计算消费总金额和消费总频次
consume_counts=card_total_consume.groupby(['user_id'])['time'].count().reset_index().sort_values("time",ascending=False) 
consume_counts.columns=['user_id','consume_counts']
consume_counts.head()


# In[135]:


consume_counts.consume_counts.hist(bins=100)


# In[134]:


#计算一年的消费总金额
consume_sum=card_total_consume.groupby(['user_id'])['amount'].sum().reset_index().sort_values("amount",ascending=False)
consume_sum.head()


# In[137]:


students_consume=pd.merge(consume_counts,consume_sum,how='inner',on='user_id')
students_consume.head()


# In[138]:


##k-means 算法聚类
from scipy.cluster.vq import vq, kmeans, whiten
from sklearn.cluster import KMeans


# In[139]:


plt.scatter(students_consume['consume_counts'],students_consume['amount'],color='blue',marker='+',linewidth=2,alpha=0.8)


# In[140]:


#设置要进行聚类的字段
consume= np.array(students_consume[['consume_counts','amount']])
#设置类别为5
clf=KMeans(n_clusters=5)
#将数据代入到聚类模型中
clf=clf.fit(consume)
#查看聚类结果
clf.cluster_centers_


# In[141]:


#在原始数据表中增加聚类结果标签
students_consume['label']=clf.labels_


# In[142]:


students_consume.head()


# In[143]:


##用散点图直观表示
students_consume0 = students_consume.loc[students_consume["label"] == 0]
students_consume1 = students_consume.loc[students_consume["label"] == 1]
students_consume2 = students_consume.loc[students_consume["label"] == 2]
students_consume3 = students_consume.loc[students_consume["label"] == 3]
students_consume4 = students_consume.loc[students_consume["label"] == 4]

plt.scatter(students_consume0['consume_counts'],students_consume0['amount'],50,color='r',marker='+',linewidth=2,alpha=0.8)
plt.scatter(students_consume1['consume_counts'],students_consume1['amount'],50,color='g',marker='+',linewidth=2,alpha=0.8)
plt.scatter(students_consume2['consume_counts'],students_consume2['amount'],50,color='b',marker='+',linewidth=2,alpha=0.8)
plt.scatter(students_consume3['consume_counts'],students_consume3['amount'],50,color='c',marker='+',linewidth=2,alpha=0.8)
plt.scatter(students_consume4['consume_counts'],students_consume4['amount'],50,color='k',marker='+',linewidth=2,alpha=0.8)


# In[144]:


students_consume.label.value_counts()


# In[145]:


students_consume.label.value_counts().plot.pie(labels=['0','4','1','3','2'],
                      autopct='%.2f',fontsize=10, figsize=(6, 6),subplots=True)
plt.axis('equal')  #避免压缩成椭圆
plt.legend()


# In[146]:


students_consume['label']=students_consume['label'].replace([0,1,2,3,4],[33,35,31,34,32])


# In[148]:


students_consume.to_csv('E:/BaiduNetdiskDownload/jzzz/students_consume.csv',sep=',', header=True,index=False)


# In[170]:


# 对生活习惯数据进行处理
card_total['hour']=card_total['time'].str[11:13]
card_total['hour']=card_total.hour.astype('int')


# In[171]:


###早餐,午餐,晚餐
card_canteen=card_total[card_total['kind']=='食堂']
card_canteen=card_canteen[['user_id','hour','kind']]


# In[172]:


card_canteen.head()


# In[173]:


card_breakfast=card_canteen[(card_canteen['hour']>=5)&(card_canteen['hour']<=8)]
card_breakfast=card_breakfast.replace('食堂','早餐')
card_lunch=card_canteen[(card_canteen['hour']>=11)&(card_canteen['hour']<=13)]
card_lunch=card_lunch.replace('食堂','午餐')
card_dinner=card_canteen[(card_canteen['hour']>=17)&(card_canteen['hour']<=19)]
card_dinner=card_dinner.replace('食堂','晚餐')


# In[174]:


card_canteen2=card_breakfast.append(card_lunch)
card_canteen2=card_canteen2.append(card_dinner)


# In[175]:


card_canteen_group=card_canteen2.groupby(['user_id','kind'])['hour'].count()


# In[177]:


card_canteen_group=card_canteen_group.unstack('kind')# 调换行列
card_canteen_group.fillna(0,inplace=True)    #用0来填补缺失值
card_canteen_group=card_canteen_group.astype('int')
card_canteen_group=card_canteen_group.reset_index()


# In[178]:


card_canteen_group.head()


# In[ ]:


score_breakfast = pd.merge(card_canteen_group,score_total, how='inner',on='user_id')


# In[187]:


plt.scatter(score_breakfast['早餐'],score_breakfast['rank_per'])


# In[189]:


##计算学生洗浴频次
card_shower=card_total[card_total['kind']=='淋浴']
card_shower_group=card_shower.groupby(['user_id'])['time'].count().reset_index()#.sort_values("book",ascending=False)  
card_shower_group.columns = ['user_id','淋浴'] 


# In[190]:


card_shower_group.head()


# In[191]:


##计算学生打水次数
card_water=card_total[card_total['kind']=='开水']
card__water_group=card_water.groupby(['user_id'])['time'].count().reset_index()
card__water_group.columns = ['user_id','开水'] 


# In[192]:


card__water_group.head()


# In[193]:


card_life=pd.merge(card_canteen_group,card_shower_group,how='outer',on='user_id')
card_life=pd.merge(card_life,card__water_group,how='outer',on='user_id')


# In[194]:


card_life.fillna(0,inplace=True)  
card_life=card_life.astype('int')
order=['user_id', '早餐', '午餐', '晚餐','淋浴', '开水']
card_life=card_life[order]


# In[195]:


card_life.head()


# In[196]:


#生活规律数据处理完毕，对其进行聚类
#设置要进行聚类的字段
life= np.array(card_life[['早餐', '午餐', '晚餐', '淋浴', '开水']])
#设置类别为6
clf=KMeans(n_clusters=6)
#将数据代入到聚类模型中
clf=clf.fit(life)
np.set_printoptions(suppress=True)   ## 取消科学记数法显示
clf.cluster_centers_


# In[198]:


card_life['label']=clf.labels_
card_life.head()


# In[199]:


card_life.label.value_counts()


# In[200]:


card_life.label.value_counts().plot.pie(labels=['0','5','3','2','1','4'],
                      autopct='%.2f',fontsize=10, figsize=(6, 6),subplots=True)
plt.axis('equal')  #避免压缩成椭圆
plt.legend()


# In[201]:


card_life['label']=card_life['label'].replace([0,1,2,3,4,5],[24,25,21,26,23,22])


# In[203]:


card_life.to_csv('E:/BaiduNetdiskDownload/jzzz/card_life.csv',sep=',', header=True,index=False)


# In[206]:


##学习努力程度数据
#训练集
filepath='E:/BaiduNetdiskDownload/jzzz/train/borrow_train.txt'
borrow_train=pd.read_table(filepath, sep = ',',   
                          names = ['user_id', 'date', 'book', 'isbn'],
                          encoding='utf-8',header=None)
#存在重复值，去重
borrow_train=borrow_train.sort_values(by=['user_id'],ascending=True)
borrow_train.drop_duplicates(['user_id','date','book'],keep='last',inplace=True)
borrow_train.head()


# In[207]:


#测试集
borrow_test=pd.read_table(r'E:\BaiduNetdiskDownload\jzzz\test\borrow_test.txt', sep = ',',   
                          names = ['user_id', 'date', 'book', 'isbn'],
                          encoding='utf-8',header=None)
borrow_test=borrow_test.sort_values(by=['user_id'],ascending=True)
borrow_test.drop_duplicates(['user_id','date','book'],keep='last',inplace=True)
borrow_test.head()


# In[208]:


borrow_total=borrow_train.append(borrow_test)
borrow_total=borrow_total.sort_values(by=['user_id'],ascending=True)
borrow_total.head()


# In[209]:


borrow_group=borrow_total.groupby(["user_id"])["book"].count().reset_index().sort_values("book",ascending=False)  
borrow_group.head()


# In[210]:


#处理图书馆的出入时间数据
library_train=pd.read_table(r'E:/BaiduNetdiskDownload/jzzz/train/library_train.txt', sep = ',', quotechar =',',
                          names = ['user_id','door','time',],
                          encoding='utf-8',header=None)
library_test=pd.read_table(r'E:/BaiduNetdiskDownload/jzzz/test/library_test.txt', sep = ',', quotechar =',',
                          names = ['user_id','door','time',],
                          encoding='utf-8',header=None)


# In[212]:


library_total=library_train.append(library_test)
library_group=library_total.groupby(['user_id'])['time'].count().reset_index()


# In[213]:


library_group.head()


# In[214]:


library_borrow=pd.merge(borrow_group,library_group,how='outer',on='user_id').sort_values(['user_id']).reset_index(drop=True)
library_borrow.fillna(0,inplace=True)  
library_borrow=library_borrow.astype('int')


# In[215]:


library_borrow.head()  


# In[216]:


#设置要进行聚类的字段
study= np.array(library_borrow[['book','time']])
#设置类别为5
clf=KMeans(n_clusters=5)
#将数据代入到聚类模型中
clf=clf.fit(study)
np.set_printoptions(suppress=True)   ## 取消科学记数法显示
clf.cluster_centers_


# In[218]:


library_borrow['label']=clf.labels_


# In[219]:


library_borrow.head()


# In[220]:


library_borrow.label.value_counts()


# In[221]:


library_borrow.label.value_counts().plot.pie(labels=['0','4','2','1','3'],
                      autopct='%.2f',fontsize=10, figsize=(6, 6),subplots=True)
plt.axis('equal')  #避免压缩成椭圆
plt.legend()


# In[222]:


library_borrow['label']=library_borrow['label'].replace([0,1,2,3,4],[45,42,43,41,44])


# In[224]:


library_borrow.to_csv('E:/BaiduNetdiskDownload/jzzz/library_borrow.csv',sep=',', header=True,index=False)


# In[225]:


students_consume.rename(columns={'label': 'consume_index'},inplace=True)    
card_life.rename(columns={'label': 'life_index'},inplace=True) 
library_borrow.rename(columns={'label': 'study_index'},inplace=True) 
students_consume_index = students_consume[['user_id','consume_index']]
card_life1_index = card_life[['user_id','life_index']]
library_borrow_index = library_borrow[['user_id','study_index']]


# In[226]:


college_relate=pd.merge(students_consume_index ,card_life1_index,how='inner',on='user_id')
college_relate=pd.merge(college_relate,library_borrow_index,how='inner',on='user_id')


# In[231]:


score_index = score_total[['user_id','label']]


# In[233]:


college_relate=pd.merge(college_relate,score_index,how='inner',on='user_id').sort_values(['user_id']).reset_index(drop=True)


# In[235]:


del college_relate['user_id']
order = ['label','life_index','consume_index','study_index']
college_relate = college_relate[order]


# In[236]:


college_relate.head()


# In[237]:


college_relate=college_relate.astype('str')
college_relate.to_csv('E:/BaiduNetdiskDownload/jzzz/college_relate3.csv',sep=',', header=False,index=False)
###版本3


# In[238]:


###apriori算法
from __future__ import print_function


# In[239]:


def connect_string(x, ms):
    x = list(map(lambda i: sorted(i.split(ms)), x))
    l = len(x[0])
    r = []
    for i in range(len(x)):
        for j in range(i, len(x)):
            if x[i][:l - 1] == x[j][:l - 1] and x[i][l - 1] != x[j][l - 1]:
                r.append(x[i][:l - 1] + sorted([x[j][l - 1], x[i][l - 1]]))
    return r

def find_rule(d, support, confidence, ms=u'--'):
    result = pd.DataFrame(index=['support', 'confidence'])  # 定义输出结果

    support_series = 1.0 * d.sum() / len(d)  # 支持度序列
    column = list(support_series[support_series > support].index)  # 初步根据支持度筛选
    k = 0

    while len(column) > 1:
        k = k + 1
        print(u'\n正在进行第%s次搜索...' % k)
        column = connect_string(column, ms)
        print(u'数目：%s...' % len(column))
        sf = lambda i: d[i].prod(axis=1, numeric_only=True)  # 新一批支持度的计算函数

        # 创建连接数据，这一步耗时、耗内存最严重。当数据集较大时，可以考虑并行运算优化。
        d_2 = pd.DataFrame(list(map(sf, column)), index=[ms.join(i) for i in column]).T

        support_series_2 = 1.0 * d_2[[ms.join(i) for i in column]].sum() / len(d)  # 计算连接后的支持度
        column = list(support_series_2[support_series_2 > support].index)  # 新一轮支持度筛选
        support_series = support_series.append(support_series_2)
        column2 = []

        for i in column:  # 遍历可能的推理，如{A,B,C}究竟是A+B-->C还是B+C-->A还是C+A-->B？
            i = i.split(ms)
            for j in range(len(i)):
                column2.append(i[:j] + i[j + 1:] + i[j:j + 1])

        cofidence_series = pd.Series(index=[ms.join(i) for i in column2])  # 定义置信度序列

        for i in column2:  # 计算置信度序列
            cofidence_series[ms.join(i)] = support_series[ms.join(sorted(i))] / support_series[ms.join(i[:len(i) - 1])]

        for i in cofidence_series[cofidence_series > confidence].index:  # 置信度筛选
            result[i] = 0.0
            result[i]['confidence'] = cofidence_series[i]
            result[i]['support'] = support_series[ms.join(sorted(i.split(ms)))]

    result = result.T.sort_values(['confidence', 'support'], ascending=False)  # 结果整理，输出
    print(u'\n结果为：')
    print(result)

    return result


# In[240]:


inputfile = 'E:/BaiduNetdiskDownload/jzzz/college_relate3.csv' #输入事务集文件
data = pd.read_csv(inputfile, header=None, dtype = object)
start = time.clock() #计时开始
print(u'\n转换原始数据至0-1矩阵...')
ct = lambda x : pd.Series(1, index = x[pd.notnull(x)]) #转换0-1矩阵的过渡函数
b = map(ct, data.as_matrix()) #用map方式执行
c = list(b)
data = pd.DataFrame(c).fillna(0) #实现矩阵转换，空值用0填充
end = time.clock() #计时结束
print(u'\n转换完毕，用时：%0.2f秒' %(end-start))
del b #删除中间变量b，节省内存

support = 0.01 #最小支持度
confidence = 0.6#最小置信度
ms = '---' #连接符，默认'--'，用来区分不同元素，如A--B。需要保证原始表格中不含有该字符
start = time.clock() #计时开始
print(u'\n开始搜索关联规则...')
find_rule(data, support, confidence, ms)
end = time.clock() #计时结束
print(u'\n搜索完成，用时：%0.2f秒' %(end-start))


# In[ ]:




