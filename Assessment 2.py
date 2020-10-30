'''
COMP809 – Data Mining and Machine Learning Assessment 2

-Due: Friday 30 October at 12 midnight
-Weighting: 50%
-Note: This assignment must be completed individually.
-Submission: A soft copy needs to be submitted through Turnitin.
             When submitting the assignment make sure your name and student number is indicated on the front page of the report.

You have been employed as a data scientist by a large data analytics company and your first project has gone well.
Your first project involved supervised learning and you were able to apply the methods that you covered in Assessment 1 of this course.
However, the second project that you have been assigned involves datasets where labels are not relevant to the problem at hand
or they involve too much time commitment from domain specialists to label them.
You realise the only method of solution is to apply unsupervised learning, specifically clustering.
As you have been assigned datasets from four very different application environments you have decided that
the best approach is to explore three widely used clustering algorithms and deploy each of them on the different datasets.

The three algorithms that you have decided to explore are 1) K Means 2) DBSCAN and 3) Agglomerative.
The four datasets that you have been given are:
1) Dow Jones Index                      https://archive.ics.uci.edu/ml/datasets/Dow+Jones+Index#
    Data Set Information:

    In predicting stock prices you collect data over some period of time - day, week, month, etc. But you cannot take advantage of data from a time period until the next increment of the time period.
    For example, assume you collect data daily. When Monday is over you have all of the data for that day. However you can invest on Monday, because you don't get the data until the end of the day.
    You can use the data from Monday to invest on Tuesday.In our research each record (row) is data for a week. Each record also has the percentage of return that stock has in the following week
    (percent_change_next_weeks_price). Ideally, you want to determine which stock will produce the greatest rate of return in the following week. This can help you train and test your algorithm.
    Some of these attributes might not be use used in your research. They were originally added to our database to perform calculations.
    (Brown, Pelosi & Dirska, 2013) used percent_change_price, percent_change_volume_over_last_wk, days_to_next_dividend, and percent_return_next_dividend. We left the other attributes in the dataset in case you wanted to use any of them. Of course what you want to maximize is percent_change_next_weeks_price.

    Training data vs Test data: In (Brown, Pelosi & Dirska, 2013) we used quarter 1 (Jan-Mar) data for training and quarter 2 (Apr-Jun) data for testing.
    Interesting data points:If you use quarter 2 data for testing, you will notice something interesting in the week ending 5/27/2011 every Dow Jones Index stock lost money.


    Attribute Information:

            quarter: the yearly quarter (1 = Jan-Mar; 2 = Apr=Jun).
            stock: the stock symbol (see above)
            date: the last business day of the work (this is typically a Friday)
            open: the price of the stock at the beginning of the week
            high: the highest price of the stock during the week
            low: the lowest price of the stock during the week
            close: the price of the stock at the end of the week
            volume: the number of shares of stock that traded hands in the week
            percent_change_price: the percentage change in price throughout the week
            percent_chagne_volume_over_last_wek: the percentage change in the number of shares of
            stock that traded hands for this week compared to the previous week
            previous_weeks_volume: the number of shares of stock that traded hands in the previous week
            next_weeks_open: the opening price of the stock in the following week
            next_weeks_close: the closing price of the stock in the following week
            percent_change_next_weeks_price: the percentage change in price of the stock in the
            following week days_to_next_dividend: the number of days until the next dividend
            percent_return_next_dividend: the percentage of return on the next dividend

2) Facebook Live Sellers in Thailand    https://archive.ics.uci.edu/ml/datasets/Facebook+Live+Sellers+in+Thailand
    Data Set Information:

    The variability of consumer engagement is analysed through a Principal Component Analysis, highlighting the changes induced by the use of Facebook Live.
    The seasonal component is analysed through a study of the averages of the different engagement metrics for different time-frames (hourly, daily and monthly).
    Finally, we identify statistical outlier posts, that are qualitatively analyzed further, in terms of their selling approach and activities.


    Attribute Information:

    status_id
    status_type
    status_published
    num_reactions
    num_comments
    num_shares
    num_likes
    num_loves
    num_wows
    num_hahas
    num_sads
    num_angrys

3) Sales Transactions                   https://archive.ics.uci.edu/ml/datasets/Sales_Transactions_Dataset_Weekly
   Abstract: Contains weekly purchased quantities of 800 over products over 52 weeks. Normalised values are provided too.
4) Water Treatment Plant                https://archive.ics.uci.edu/ml/datasets/Water+Treatment+Plant
You will need to complete three tasks as detailed below.
'''


import time
import numpy             as np
import pandas            as pd
import matplotlib.cm     as cm
import matplotlib.pyplot as plt


from sklearn.datasets       import make_blobs
from sklearn.cluster        import KMeans
from sklearn.cluster        import AgglomerativeClustering
from sklearn.cluster        import DBSCAN
from sklearn.metrics        import silhouette_score
from sklearn.metrics        import silhouette_samples
from sklearn.metrics        import davies_bouldin_score
from sklearn                import preprocessing
from sklearn.decomposition  import PCA
from sklearn.manifold       import TSNE
from sklearn.datasets       import load_iris,load_digits
from sklearn.preprocessing  import MinMaxScaler
from collections            import Counter




File_Path_DataSet   = r'/Users/chunjiangmei/Documents/OneDrive - AUT University/Semester 2 2020/COMP809_Data Mining and Machine Learning/Assessment2/'
Dataset_Name        = ['Dow Jones Index','Facebook Live Sellers in Thailand','Sales Transaction','Water Treatment Plant']
Dataset_filename    = [ 'Dataset1_dow_jones_index.csv', 'Dataset2_Facebook_Live_Sellers_in_Thailand.csv', 'Dataset3_Sales_Transactions_Dataset_Weekly.csv', 'Dataset4_water_treatment.csv']

n_Clusters      = [2,3,4,5,6,7,8,9,10]
G_X_Dataset     = [[], [], [], []]
G_y_Dataset     = [[], [], [], []]

G_TIME_Dataset  = [[], [], [], []]
G_SSE_Dataset   = [[], [], [], []]
G_CSM_Dataset   = [[], [], [], []]
G_DBI_Dataset   = [[], [], [], []]

PCA_n_components = 3
#PCA_n_components = 'mle'

'''
K means Cluster =  10
SSE of Dataset 1 is  [1.3226720595532993e+19, 5.941134760170267e+18, 4.570699816132532e+18, 3.599399862662879e+18, 3.083106086980572e+18, 2.6287698036167834e+18, 2.2943930933739005e+18, 2.0251146052845711e+18, 1.7459218187805514e+18]
SSE of Dataset 2 is  [4986367687.004714, 2735233726.7572927, 2002121450.7013447, 1469911698.3619347, 1186897917.9479122, 923442874.3298947, 749949526.0243564, 621397403.6057582, 523170241.3671376]
SSE of Dataset 3 is  [1089089.6427620265, 349356.5103417699, 231793.7801367796, 162614.21940463173, 144859.83537437435, 129132.94702121934, 118029.17823860046, 107963.38055311632, 103448.18836481195]
SSE of Dataset 4 is  [113827202.41718265, 77300499.67184113, 57492356.103866346, 49693185.957826845, 43866479.048925206, 38698046.64990924, 34764518.67114883, 31814829.64796366, 29651024.488785893]

SSE of Dataset 1 is  [4381753.927929005, 3437905.795422118, 2502616.6767444005, 2054981.8812749584, 1652000.345877402, 1488533.7692206975, 1336415.0938543177, 1225175.5316829914, 1138171.7439224988]
SSE of Dataset 2 is  [4986367687.004707, 2735233726.7572885, 2002121450.701343, 1469911698.3619337, 1186897917.9479105, 923442874.3298937, 749949526.0243534, 621397403.6057575, 523170241.3671367]
SSE of Dataset 3 is  [1089141.3680440788, 349377.6159422226, 231806.63825198112, 162611.55222531687, 144899.8487008755, 129121.49014549547, 118043.2530304422, 111671.94312064309, 103579.33256429581]
SSE of Dataset 4 is  [113827202.41718265, 77300499.67184113, 57492356.103866346, 49693185.957826845, 43866479.048925206, 38698046.64990924, 34764518.67114883, 31814829.64796366, 29651024.488785893]
'''

def DataPreProcessing():
    print('\n#[Log]Raw data Preprocessing () # Feature Selection by PCA Algorithm for all Datasets! ' )
    for i in range(0,4):
        rawdata = pd.read_csv(File_Path_DataSet + Dataset_filename[i])
        print(' -- For Dataset' + np.str(i + 1) + ':Raw data shape is ', rawdata.shape)
        #print(rawdata.describe())
        if i==0:                                            # For Dataset1_dow_jones_index
            # pre-processing the quarter feature

            # pre-processing the stock feature
            # pre-processing the date feature
            # pre-processing the NAN
            stockname = ['AA','AXP','BA','BAC','CAT','CSCO','CVX','DD','DIS','GE','HD','HPQ','IBM','INTC','JNJ','JPM','KRFT','KO','MCD','MMM','MRK','MSFT','PFE','PG','T','TRV','UTX','VZ','WMT','XOM']
            rawdata = rawdata.drop(['quarter','stock','date'],axis=1) # 不删除'volume','previous_weeks_volume' 这两个值，得到的SSE肘更新明显
            rawdata = rawdata.dropna()

            X = rawdata.values
        elif i==1:                                           # For Dataset2_Facebook_Live_Sellers_in_Thailand
            array = rawdata.values
            X = array[:, 3:11] # 舍弃前3列string数据+后面四列Null列 ，只用from num_reactions to num_angrys
        elif i==2:                                           # For Dataset3_Sales_Transactions_Dataset_Weekly
            nrow, ncol = rawdata.shape
            array = rawdata.values
            X = array[:, 1:53] # 舍弃前1列string数据
        elif i==3:                                           # For Dataset4_water_treatment
            df = pd.DataFrame(rawdata.values)
            df.replace('?', np.NAN, inplace=True)
            rawdata = df.dropna()  # Dropna
            array = rawdata.values
            X = array[:, 2:nrow] # 舍弃前1列数据


        X = preprocessing.StandardScaler().fit_transform(X)
        X = preprocessing.Normalizer().fit_transform(X)
        pca = PCA(PCA_n_components)
        G_X_Dataset[i] = pca.fit_transform(X)

        print(G_X_Dataset[i] .mean(axis=0))
        print('    For Dataset'+np.str(i+1)+':Preprocessed data shape   is ',G_X_Dataset[i].shape )
        print(pd.DataFrame(G_X_Dataset[i]).describe())

        # 可尝试将数据经过PCA降纬度后 现实出来

        #print(G_X_Dataset[i])len(G_X_Dataset[i])
        #print(pca.n_components)
        #print(pca.explained_variance_ratio_)    # 占所有特征的方差百分比 越大越重要
        #print(pca.explained_variance_)          # 特征的方差数值       越大越重要

def test_params_DBSCAN(dataset):
    print('Dataset is ',np.int(dataset+1))
    i_esp = np.float(0.1)
    while i_esp < 5:
        #for i_samples in range(10, 20):
        algo = DBSCAN(i_esp, min_samples=np.int(2 * PCA_n_components))  # min_samples = 2*D
        y_pred = algo.fit_predict(G_X_Dataset[dataset])
        print('esp = ',  i_esp,  '\t\t', 'min_sample = ', np.int(2 * PCA_n_components), '\t\t', Counter(y_pred), '\t\t', 'score = ',silhouette_score(G_X_Dataset[dataset], y_pred,
                                                                                                                                                              metric='euclidean'))
        i_esp = i_esp + np.float(0.1)
        #print('\n')
    '''
    # Database 1 : esp =  1.2 		 min_sample =  6 		 Counter({0: 688, 1: 115, -1: 8}) 		 score =  0.6001765597691359
    # Database 2 : 
    # Database 3 :
    # Database 4 :
    '''

def Implement_Algorithm(algorithm):
    global G_X_Dataset,G_y_Dataset,G_TIME_Dataset,G_SSE_Dataset,G_CSM_Dataset,G_DBI_Dataset
    print("\n#Implement_Algorithm( "+ algorithm+" )")

    if   algorithm == "K means":
        for k in n_Clusters:
            print(algorithm, 'Cluster = ', k)
            for i in range(0, 4):
                starttime = time.time()
                algo = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=len(G_X_Dataset[i]), random_state=0)
                y_pred = algo.fit_predict(G_X_Dataset[i])
                endtime = time.time()
                G_TIME_Dataset[i].append(endtime - starttime)                                                       # Time Taken
                G_SSE_Dataset[i].append(algo.inertia_)                                                              # SSE
                score = silhouette_score(G_X_Dataset[i], y_pred, metric='euclidean')
                G_CSM_Dataset[i].append(score )              # CSM
                #print('Time Taken of Dataset', i+1, 'is ', G_TIME_Dataset[i])
                #print('SSE of Dataset', i+1, 'is ', G_SSE_Dataset[i])
                #print('CSM of Dataset', i+1, 'is ', G_CSM_Dataset[i])

    elif algorithm == "DBSCAN":
        #test_params_DBSCAN(np.int(0))
        #test_params_DBSCAN(np.int(1))
        #test_params_DBSCAN(np.int(2))
        #test_params_DBSCAN(np.int(3))
        for k in n_Clusters:
            for i in range(0, 4):
                starttime = time.time()
                if   i==0: algo = DBSCAN(eps=1.2, min_samples=6)
                elif i==1: algo = DBSCAN(eps=1.2, min_samples=6)
                elif i==2: algo = DBSCAN(eps=1.2, min_samples=6)
                elif i==3: algo = DBSCAN(eps=1.2, min_samples=6)
                y_pred = algo.fit_predict(G_X_Dataset[i])
                endtime = time.time()
                if len(Counter(y_pred)) == 1:
                    print('Error!' , Counter(y_pred) )
                    break
                G_TIME_Dataset[i].append(endtime - starttime)                                           # Time Taken
                G_DBI_Dataset[i].append(davies_bouldin_score(G_X_Dataset[i],y_pred))                    # DBI
                G_CSM_Dataset[i].append(silhouette_score(G_X_Dataset[i], y_pred, metric='euclidean'))   # CSM
            print('Time Taken of Dataset', i+1, 'is ', G_TIME_Dataset[i])
            print('DBI of Dataset', i+1, 'is ', G_DBI_Dataset[i])
            print('CSM of Dataset', i+1, 'is ', G_CSM_Dataset[i])

    elif algorithm == "Agglomerative":
        for k in n_Clusters:
            print(algorithm, 'Cluster = ', k)
            for i in range(0, 4):
                starttime = time.time()
                algo = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='complete')
                y_pred = algo.fit_predict(G_X_Dataset[i])
                endtime = time.time()
                if len(Counter(y_pred)) == 1:
                    print('Error!' , Counter(y_pred) )
                    break
                G_TIME_Dataset[i].append(endtime - starttime)                                                       # Time Taken
                G_DBI_Dataset[i].append(davies_bouldin_score(G_X_Dataset[i],y_pred))                                # DBI
                G_CSM_Dataset[i].append(silhouette_score(G_X_Dataset[i], y_pred, metric='euclidean'))               # CSM
            print('Time Taken of Dataset', i+1, 'is ', G_TIME_Dataset[i])
            print('SSE of Dataset', i+1, 'is ', G_DBI_Dataset[i])
            print('CSM of Dataset', i+1, 'is ', G_CSM_Dataset[i])

    else: print("#[Algorithm_error]")

def Display_Measure_Result( algorithm ):
    print("#Display Measure Results")
    global G_X_Dataset, G_y_Dataset, G_TIME_Dataset, G_SSE_Dataset, G_CSM_Dataset
    fig,axe = plt.subplots(4,3)
    fig.set_size_inches(20, 13)
    # Setting Figure title, xlabel, ylabel
    for i in range(0,4):
        axe[i][0].set_title ("The TimeTaken of Dataset [ "+ Dataset_Name[i]+' ]')
        axe[i][0].set_xlabel("Number of clusters")
        axe[i][0].set_ylabel("time ")
        axe[i][1].set_title ("SSE of Dataset [ "+ Dataset_Name[i]+' ]')
        axe[i][1].set_xlabel("Number of clusters")
        axe[i][1].set_ylabel("Distortion ")
        axe[i][2].set_title ("CSM of Dataset [ "+ Dataset_Name[i]+' ]')
        axe[i][2].set_xlabel("Number of clusters")
        axe[i][2].set_ylabel("CSM Scores")
    if algorithm == "K means":
        plt.suptitle("Algorithm : K Means",       fontsize=18, fontweight='bold')
    elif algorithm == "DBSCAN":
        plt.suptitle("Algorithm : DBSCAN",        fontsize=18, fontweight='bold')
    elif algorithm == "Agglomerative":
        plt.suptitle("Algorithm : Agglomerative", fontsize=18, fontweight='bold')
    else:print("//[Algorithm_error]")

    for i in range(0,4):
        axe[i][0].plot(n_Clusters, G_TIME_Dataset[i], marker='o')
        axe[i][1].plot(n_Clusters, G_SSE_Dataset[i], marker='o')
        #axe[i][1].plot(n_Clusters, G_DBI_Dataset[i], marker='o')
        axe[i][2].plot(n_Clusters, G_CSM_Dataset[i], marker='o')
    fig.tight_layout()
    plt.savefig(File_Path_DataSet+'Result - '+algorithm, dpi=300)
    # plt.show()

def Display_CSM_plot(algorithm):
    print('\n#Display the CSM plot for the best value of the K parameter for each dataset')
    global G_X_Dataset

    # 【注意】需要根据不同算法的最佳K值来设定
    n_clusters = 3                                  # 先设定我们要分成的簇数 先用通用值 3
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)  # 创建一个画布，画布上共有一行四列四个图
    fig.set_size_inches(17, 10)                     # 画布尺寸
    plt.suptitle("Algorithm : " + algorithm , fontsize=18, fontweight='bold')
    # 第一个图是我们的轮廓系数图像，是由各个簇的轮廓系数组成的横向条形图
    # 横向条形图的横坐标是我们的轮廓系数取值，纵坐标是我们的每个样本，因为轮廓系数是对于每一个样本进行计算的
    # 首先我们来设定横坐标
    # 轮廓系数的取值范围在[-1,1]之间，但我们至少是希望轮廓系数要大于0的
    # 太长的横坐标不利于我们的可视化，所以只设定X轴的取值在[-0.1,1]之间
    ax1.set_xlim([0, 1])
    ax2.set_xlim([0, 1])
    ax3.set_xlim([0, 1])
    ax4.set_xlim([0, 1])

    # 接下来设定纵坐标，通常来说，纵坐标是从0开始，最大值取到X.shape[0]的取值
    # 但我们希望，每个簇能够排在一起，不同的簇之间能够有一定的空隙
    # 以便我们看到不同的条形图聚合成的块，理解它是对应了哪一个簇
    # 因此我们在设定纵坐标的取值范围的时候，在X.shape[0]上，加上一个距离(n_clusters + 1) * 10，留作间隔用
    ax1.set_ylim([0, G_X_Dataset[0].shape[0] + (n_clusters + 1) * 10])
    ax2.set_ylim([0, G_X_Dataset[1].shape[0] + (n_clusters + 1) * 10])
    ax3.set_ylim([0, G_X_Dataset[2].shape[0] + (n_clusters + 1) * 10])
    ax4.set_ylim([0, G_X_Dataset[3].shape[0] + (n_clusters + 1) * 10])


    # 开始建模，调用聚类好的标签

    for i in range(0,4):
        if   algorithm == "K means" :
            n_clusters = 3
            clusterer_Dataset1 = KMeans(n_clusters=n_clusters, random_state=10).fit(G_X_Dataset[0])
            clusterer_Dataset2 = KMeans(n_clusters=n_clusters, random_state=10).fit(G_X_Dataset[1])
            clusterer_Dataset3 = KMeans(n_clusters=n_clusters, random_state=10).fit(G_X_Dataset[2])
            clusterer_Dataset4 = KMeans(n_clusters=n_clusters, random_state=10).fit(G_X_Dataset[3])
        elif algorithm == "DBSCAN"  :
            print('DBSCAN')
            # n_clusters = 4
            clusterer_Dataset1 = DBSCAN(1.2, min_samples=np.int(2 * PCA_n_components)).fit(G_X_Dataset[0])
            clusterer_Dataset2 = DBSCAN(1.2, min_samples=np.int(2 * PCA_n_components)).fit(G_X_Dataset[1])
            clusterer_Dataset3 = DBSCAN(1.2, min_samples=np.int(2 * PCA_n_components)).fit(G_X_Dataset[2])
            clusterer_Dataset4 = DBSCAN(1.2, min_samples=np.int(2 * PCA_n_components)).fit(G_X_Dataset[3])
        elif algorithm == "Agglomerative" :
            print('Agglomerative')
            n_clusters = 5
            clusterer_Dataset1 = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='complete').fit(G_X_Dataset[0])
            clusterer_Dataset2 = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='complete').fit(G_X_Dataset[1])
            clusterer_Dataset3 = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='complete').fit(G_X_Dataset[2])
            clusterer_Dataset4 = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='complete').fit(G_X_Dataset[3])
        else : print("algorithm error!")

        cluster_labels_Dataset1 = clusterer_Dataset1.labels_
        cluster_labels_Dataset2 = clusterer_Dataset2.labels_
        cluster_labels_Dataset3 = clusterer_Dataset3.labels_
        cluster_labels_Dataset4 = clusterer_Dataset4.labels_


        #print(G_X_Dataset[0].shape[0],G_X_Dataset[1].shape[0],G_X_Dataset[2].shape[0],G_X_Dataset[3].shape[0])
        #print(cluster_labels_Dataset1,cluster_labels_Dataset2,cluster_labels_Dataset3,cluster_labels_Dataset4)

        # 调用轮廓系数分数，注意，silhouette_score生成的是所有样本点的轮廓系数均值
        # 两个需要输入的参数是，特征矩阵X和聚类完毕后的标签
        silhouette_avg_Dataset1 = silhouette_score(G_X_Dataset[0], cluster_labels_Dataset1)
        silhouette_avg_Dataset2 = silhouette_score(G_X_Dataset[1], cluster_labels_Dataset2)
        silhouette_avg_Dataset3 = silhouette_score(G_X_Dataset[2], cluster_labels_Dataset3)
        silhouette_avg_Dataset4 = silhouette_score(G_X_Dataset[3], cluster_labels_Dataset4)

        # 用print来报一下结果，现在的簇数量下，整体的轮廓系数究竟有多少
        print("For n_clusters =", n_clusters, "The average silhouette_score of Dataset1 is :", silhouette_avg_Dataset1)
        print("For n_clusters =", n_clusters, "The average silhouette_score of Dataset2 is :", silhouette_avg_Dataset2)
        print("For n_clusters =", n_clusters, "The average silhouette_score of Dataset3 is :", silhouette_avg_Dataset3)
        print("For n_clusters =", n_clusters, "The average silhouette_score of Dataset4 is :", silhouette_avg_Dataset4)

        # 调用silhouette_samples，返回每个样本点的轮廓系数，这就是我们的横坐标
        sample_silhouette_values_Dataset1 = silhouette_samples(G_X_Dataset[0], cluster_labels_Dataset1)
        sample_silhouette_values_Dataset2 = silhouette_samples(G_X_Dataset[1], cluster_labels_Dataset2)
        sample_silhouette_values_Dataset3 = silhouette_samples(G_X_Dataset[2], cluster_labels_Dataset3)
        sample_silhouette_values_Dataset4 = silhouette_samples(G_X_Dataset[3], cluster_labels_Dataset4)

        # 设定y轴上的初始取值
        y_lower_Dataset1 = 10
        y_lower_Dataset2 = 10
        y_lower_Dataset3 = 10
        y_lower_Dataset4 = 10


        # 接下来，对每一个簇进行循环
        for j in range(n_clusters):
            # 从每个样本的轮廓系数结果中抽取出第j个簇的轮廓系数，并对他进行排序

            ith_cluster_silhouette_values_Dataset1 = sample_silhouette_values_Dataset1[cluster_labels_Dataset1 == j]
            ith_cluster_silhouette_values_Dataset2 = sample_silhouette_values_Dataset2[cluster_labels_Dataset2 == j]
            ith_cluster_silhouette_values_Dataset3 = sample_silhouette_values_Dataset3[cluster_labels_Dataset3 == j]
            ith_cluster_silhouette_values_Dataset4 = sample_silhouette_values_Dataset4[cluster_labels_Dataset4 == j]

            # 注意, .sort()这个命令会直接改掉原数据的顺序
            ith_cluster_silhouette_values_Dataset1.sort()
            ith_cluster_silhouette_values_Dataset2.sort()
            ith_cluster_silhouette_values_Dataset3.sort()
            ith_cluster_silhouette_values_Dataset4.sort()

            # 查看这一个簇中究竟有多少个样本
            size_cluster_j_Dataset1 = ith_cluster_silhouette_values_Dataset1.shape[0]
            size_cluster_j_Dataset2 = ith_cluster_silhouette_values_Dataset2.shape[0]
            size_cluster_j_Dataset3 = ith_cluster_silhouette_values_Dataset3.shape[0]
            size_cluster_j_Dataset4 = ith_cluster_silhouette_values_Dataset4.shape[0]
            # 这一个簇在y轴上的取值，应该是由初始值(y_lower)开始，到初始值+加上这个簇中的样本数量结束(y_upper)
            y_upper_Dataset1 = y_lower_Dataset1 + size_cluster_j_Dataset1
            y_upper_Dataset2 = y_lower_Dataset2 + size_cluster_j_Dataset2
            y_upper_Dataset3 = y_lower_Dataset3 + size_cluster_j_Dataset3
            y_upper_Dataset4 = y_lower_Dataset4 + size_cluster_j_Dataset4

            # colormap库中的，使用小数来调用颜色的函数
            # 在nipy_spectral([输入任意小数来代表一个颜色])
            # 在这里我们希望每个簇的颜色是不同的，我们需要的颜色种类刚好是循环的个数的种类
            # 在这里，只要能够确保，每次循环生成的小数是不同的，可以使用任意方式来获取小数
            # 在这里，我是用i的浮点数除以n_clusters，在不同的i下，自然生成不同的小数
            # 以确保所有的簇会有不同的颜色
            color = cm.nipy_spectral(float(j) / n_clusters)

            # 开始填充子图1中的内容
            # fill_between是填充曲线与直角之间的空间的函数
            # fill_betweenx的直角是在纵坐标上
            # fill_betweeny的直角是在横坐标上
            # fill_betweenx的参数应该输入(定义曲线的点的横坐标，定义曲线的点的纵坐标，柱状图的颜色)
            ax1.fill_betweenx(np.arange(y_lower_Dataset1, y_upper_Dataset1), ith_cluster_silhouette_values_Dataset1, facecolor=color, alpha=0.7)
            ax2.fill_betweenx(np.arange(y_lower_Dataset2, y_upper_Dataset2), ith_cluster_silhouette_values_Dataset2, facecolor=color, alpha=0.7)
            ax3.fill_betweenx(np.arange(y_lower_Dataset3, y_upper_Dataset3), ith_cluster_silhouette_values_Dataset3, facecolor=color, alpha=0.7)
            ax4.fill_betweenx(np.arange(y_lower_Dataset4, y_upper_Dataset4), ith_cluster_silhouette_values_Dataset4, facecolor=color, alpha=0.7)
            # 为每个簇的轮廓系数写上簇的编号，并且让簇的编号显示坐标轴上每个条形图的中间位置
            # text的参数为(要显示编号的位置的横坐标，要显示编号的位置的纵坐标，要显示的编号内容)
            ax1.text(-0.05, y_lower_Dataset1 + 0.5 * size_cluster_j_Dataset1, str(j))
            ax2.text(-0.05, y_lower_Dataset2 + 0.5 * size_cluster_j_Dataset2, str(j))
            ax3.text(-0.05, y_lower_Dataset3 + 0.5 * size_cluster_j_Dataset3, str(j))
            ax4.text(-0.05, y_lower_Dataset4 + 0.5 * size_cluster_j_Dataset4, str(j))
            # 为下一个簇计算新的y轴上的初始值，是每一次迭代之后，y的上线再加上10
            # 以此来保证，不同的簇的图像之间显示有空隙
            y_lower_Dataset1 = y_upper_Dataset1 + 5
            y_lower_Dataset2 = y_upper_Dataset2 + 5
            y_lower_Dataset3 = y_upper_Dataset3 + 5
            y_lower_Dataset4 = y_upper_Dataset4 + 5

    # 给图1加上标题，横坐标轴，纵坐标轴的标签
    ax1.set_title("The CSM plot for [Dow Jones Index]")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax2.set_title("The CSM plot for [Facebook Live Sellers in Thailand]")
    ax2.set_xlabel("The silhouette coefficient values")
    ax2.set_ylabel("Cluster label")
    ax3.set_title("The CSM plot for [Sales Transactions]")
    ax3.set_xlabel("The silhouette coefficient values")
    ax3.set_ylabel("Cluster label")
    ax4.set_title("The CSM plot for [Water Treatment Plant]")
    ax4.set_xlabel("The silhouette coefficient values")
    ax4.set_ylabel("Cluster label")
    # 把整个数据集上的轮廓系数的均值以虚线的形式放入我们的图中
    ax1.axvline(x=silhouette_avg_Dataset1, color="red", linestyle="--")
    ax2.axvline(x=silhouette_avg_Dataset2, color="red", linestyle="--")
    ax3.axvline(x=silhouette_avg_Dataset3, color="red", linestyle="--")
    ax4.axvline(x=silhouette_avg_Dataset4, color="red", linestyle="--")

    # 让y轴不显示任何刻度
    ax1.set_yticks([])
    ax2.set_yticks([])
    ax3.set_yticks([])
    ax4.set_yticks([])

    # 让x轴上的刻度显示为我们规定的列表
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax2.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax3.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax4.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    fig.tight_layout()
    plt.savefig(File_Path_DataSet+'Result - '+algorithm+'_CSM Plot', dpi=300)
    #plt.show()


# Task 1 (39 marks)
'''
    For each activity in this task you must apply a suitable feature selection algorithm before deploying each clustering algorithm. 
    Your clustering results should include the following measures:
    Time taken, Sum of Squares Errors (SSE), Cluster Silhouette Measure (CSM)
    Submit Python code used for parts a) to c) below. You only need to submit the code for one of the 4 datasets.
    
    a) (15 marks)
       1) Run the K means algorithm on each of the four datasets. 
       2) Obtain the best value of K using either SSE and/or CSM. 
          Tabulate your results in a 4 by 3 table, with each row corresponding to a dataset and each column corresponding to one of the three measures mentioned above. 
       3) Display the CSM plot for the best value of the K parameter for each dataset.
        
    b) (12 marks)
       Repeat the same activity for DBSCAN and tabulate your results once again, just as you did for part a). 
       Display the CSM plot and the 4 by 3 table for each dataset. 
       
    c) (12 marks)
       Finally, use the Agglomerative algorithm and document your results as you did for parts a) and b). 
       Display the CSM plot and the 4 by 3 table for each dataset. 
'''
def Task_1():

    print("###########Task 1 (a)###########")
    DataPreProcessing()

    Implement_Algorithm("K means")
    Display_Measure_Result("K means")
    Display_CSM_plot("K means")

    print("\n###########Task 1 (b)###########")
    Implement_Algorithm("DBSCAN")
    #Display_Measure_Result("DBSCAN")
    #Display_CSM_plot("DBSCAN")

    print("###########Task 1 (c)###########")
    Implement_Algorithm("Agglomerative")
    #Display_Measure_Result("Agglomerative")
    #Display_CSM_plot("Agglomerative")

# Task 2 (31 marks)
'''

    a) (12 marks)
    For each dataset identify which clustering algorithm performed best. Justify your answer.
    In the event that no single algorithm performs best on all three performance measures 
    you will need to carefully consider how you will rate each of the measures and then decide how you will produce an 
    overall measure that will enable you to rank the algorithms. 
    
    b) (12 marks)
    For each winner algorithm and for each dataset explain why it produced the best value for the CSM measure. 
    This explanation must refer directly to the conceptual design details of the algorithm. 
    There is no need to produce any further experimental evidence for this part of the question. 
    
    c) (7 marks)
    Based on what you produced in a) above, which clustering algorithm would you consider to be the overall winner 
    (i.e. after taking into consideration performance across all four datasets). Justify your answer. 
'''
def Task_2(): print("Task 2")
# Task 3 (30 marks)
'''
    This task requires you to do some further research on your own. 
    The t-sne algorithm (https://lvdmaaten.github.io/tsne/) was designed to visualize high dimensional data after reducing dimensionality.

    a) (7 marks)
    After gaining an understanding of how it works identify one important potential advantage of t-sne over Principal Components Analysis (PCA). 
    You may use one or more sources from the machine learning literature to support your answer. 
    
    b) (23 marks)
    Select the Sales Transactions dataset that you experimented with:
        (1). (5 marks)
            apply t-sne to reduce dimensionality to two components and then visualize the data using a suitable plot. 
            Submit the Python code and your plot. 
        (2). (8 marks)
            is the potential advantage of t-sne over PCA that you mentioned in part a) present itself in this dataset? 
            Justify your answer with suitable experimental evidence. 
        (3). (5 marks)
            does the 2D visualization give insights into the structure of the data? Explain your answer. 
        (4). (5 marks)
            if so, does it inform the choice of which clustering algorithm to use? On the other hand, 
            if it does not narrow down the choice of clustering algorithm then explain why the visual is insufficient on its own draw a definite conclusion. 
'''

def prepocessing_tsne(data, n):
    starttime_tsne = time.time()
    dataset = TSNE(n_components=n, random_state=33).fit_transform(data)
    endtime_tsne = time.time()
    print('cost time by tsne:', endtime_tsne - starttime_tsne)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_tsne = scaler.fit_transform(dataset)
    return X_tsne

def prepocessing_pca(data, n):
    starttime_pca = time.time()
    dataset = PCA(n_components=n).fit_transform(data)
    endtime_pca = time.time()
    print('cost time by pca:', endtime_pca - starttime_pca)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_pca = scaler.fit_transform(dataset)
    return X_pca

def sampletest():
    iris = load_iris()
    iris_data = iris.data
    digits = load_digits()
    digits_data = digits.data

    digits_tsne = prepocessing_tsne(digits_data, 2)
    digits_pca = prepocessing_pca(digits_data, 2)
    iris_tsne = prepocessing_tsne(iris_data, 2)
    iris_pca = prepocessing_pca(iris_data, 2)

    #sns.set_style("darkgrid") #设立风格

    print(digits.target)

    plt.figure(figsize=(18, 8))

    plt.subplot(1, 2, 1)
    plt.scatter(digits_tsne[:, 0], digits_tsne[:, 1], c=digits.target, alpha=0.6,
                cmap=plt.cm.get_cmap('rainbow', 10))
    plt.title("digits t-SNE", fontsize=18)
    cbar = plt.colorbar(ticks=range(10))
    cbar.set_label(label='digit value', fontsize=18)

    plt.subplot(1, 2, 2)
    plt.scatter(digits_pca[:, 0], digits_pca[:, 1], c=digits.target, alpha=0.6,
                cmap=plt.cm.get_cmap('rainbow', 10))
    plt.title("digits PCA", fontsize=18)
    cbar = plt.colorbar(ticks=range(10))
    cbar.set_label(label='digit value', fontsize=18)
    plt.tight_layout()
    #plt.show()

def Task_3():
    #data = pd.read_csv(File_Path_DataSet + Dataset_filename[2]).values[:,55:107] # get the Normalized Columns
    data = pd.read_csv(File_Path_DataSet + Dataset_filename[2]).values[:, 1:53] # get the rawdata Columns
    #print(data)

    data = preprocessing.StandardScaler().fit_transform(data)
    data = preprocessing.Normalizer().fit_transform(data)

    # algorithm on dataset
    algo = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=len(data), random_state=0)

    y_pred = algo.fit_predict(data)

    #print(y_pred)

    # Preprocessing by pca
    dataset_pca = PCA(n_components=2).fit_transform(data)
    # Preprocessing by t-SNE
    dataset_tsne = TSNE(n_components=2, random_state=33).fit_transform(data)

    print(dataset_pca)

    print(dataset_tsne)

    plt.figure(figsize=(18, 8))

    plt.subplot(1, 2, 1)
    plt.scatter(dataset_tsne[:, 0], dataset_tsne[:, 1], alpha=0.6, c=y_pred, cmap=plt.cm.get_cmap('rainbow', 10))
    plt.title("t-SNE", fontsize=18)
    cbar = plt.colorbar(ticks=range(10))
    cbar.set_label(label='value', fontsize=18)

    plt.subplot(1, 2, 2)
    plt.scatter(dataset_pca[:, 0], dataset_pca[:, 1], alpha=0.6, c=y_pred, cmap=plt.cm.get_cmap('rainbow', 10))
    plt.title(" PCA", fontsize=18)
    cbar = plt.colorbar(ticks=range(10))
    cbar.set_label(label=' value', fontsize=18)

    print(plt.cm.get_cmap('rainbow', 10))

    plt.tight_layout()
    plt.savefig(File_Path_DataSet +'Result - '+'t-SNE VS PAC', dpi=300)
    #plt.show()
def main_Assessment_2():
    Task_1()
    Task_3()

    sampletest()


main_Assessment_2()