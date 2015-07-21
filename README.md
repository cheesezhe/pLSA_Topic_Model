## PLSI Implementation ##

### How to extract features for the dataset ###
特征矩阵为t_d[V,D], V表示单词的数目，D表示文档的数目。过滤掉文档中包含的stopwords中的单词。


### Source Code Explanation ###
在源代码中，包含如下几个文件：    
1. plsa.py : plsa的具体实现和一些用到的函数，如loglikelihood的计算公式，normlize的计算。
1. pprocess.py : 对data.txt的预处理，包括stopword的过滤，构造term_document矩阵，构造word到word_id的影射，doc到doc_id的影射。
1. main.py : 程序的入口，调用plsa.py和pprocess.py，按照topic从3到D便利，每次pLSA计算的loglihood和每个topic对应的10个关键词以及对应的概率，输出到plsa.log文件中。

### Main Function Explanation ###

#### plsa.py ####

##### normalize ##### 
 对一个向量进行正则化

##### llhood ##### 
计算loglikelihood，输入t_d, p_z, p_w_z, p_d_z。 

* t_d : term_doc
* p_z : P(z)向量, topic为z时候的概率
* p_w_z : P(w|z)矩阵, topic为z时候, word为w的条件概率
* p_d_z : P(d|z)矩阵

    def llhood(t_d, p_z, p_w_z, p_d_z):
      V,D = t_d.shape # V 表示 word的个数，D表示doc的个数
      ret = 0.0
      for w,d in zip(*t_d.nonzero()):
        # 计算 sum( P(w|z) * P(d|z) * P(z) ), p_d_w是一个值，表示对给定的w,d的P(d,w)
        p_d_w = np.sum(p_z * p_w_z[w,:] * p_d_z[d,:])
        if p_d_w > 0 : 
          ret += t_d[w][d] * np.log(p_d_w)
      return ret


`pLSA.train` :

主体的迭代过程如下：

    while True :
      logging.info('[ iteration ]  step %d' %step)
      step += 1

      # 初始化 概率矩阵P(d|z), P(w|z)， 向量P(z)
      p_d_z *= 0.0
      p_w_z *= 0.0
      p_z *= 0.0

      # EM算法
      for w_idx, d_idx in zip(*t_d.nonzero()):
        #print '[ EM ] >>>>>>>>>> E step : '
        # E step 计算 P(z|d,w) , p_z_d_w是一个向量，向量中的第i个元素表示对应z = i的topic的P(z|d,w)
        p_z_d_w = pp_z * pp_d_z[d_idx,:] * pp_w_z[w_idx, :]

        normalize(p_z_d_w)
        
        #print '[ EM ] >>>>>>>>>> M step : '
        # M step , tt表示作业PPT给出的 n(d,w) * P(z|d,w)
        tt = t_d[w_idx,d_idx] * p_z_d_w
        # 更新 P(w | z)
        p_w_z[w_idx, :] += tt

        # 更新 P(d | z)
        p_d_z[d_idx, :] += tt
  
        # 更新 P(z)
        p_z += tt

      normalize(p_w_z)
      normalize(p_d_z)
      p_z = p_z / t_d.sum()

      # 计算loglikelihood，并且检查是否收敛了
      l1 = llhood(t_d, pp_z, pp_w_z, pp_d_z)
      l2 = llhood(t_d, p_z, p_w_z, p_d_z)
      
      diff = l2 - l1

      logging.info('[ iteration ] l2-l1  %.3f - %.3f = %.3f ' %(l2, l1, diff))
    
      if abs(diff) < eps :
        logging.info('[ iteration ] End EM ')
        return (l2, p_d_z, p_w_z, p_z)

      pp_d_z = p_d_z.copy()
      pp_w_z = p_w_z.copy()
      pp_z = p_z.copy()


#### pprocess.py ####

##### pprocess.get_t_d #####
对输入数据进行预处理，过滤stopword，得到term_document矩阵t_d

##### get_word #####
返回给定的word_id对应的word

#### main.py ####

##### main #####

    # Setup logging -------------------------
    logging.basicConfig(filename='plsa.log', level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
  
    # Some basic configuration ---------------
    fname = './data.txt'      # 输入数据的路径
    fsw = './stopwords.txt'   # stopword文件路径
    eps = 20.0                # 收敛的eps
    key_word_size = 10        # 每个topic输出概率最大的word的个数
  
    # Preprocess -----------------------------
    pp = PP(fname, fsw)
    t_d = pp.get_t_d()
  
    V,D = t_d.shape
    logging.info("V = %d  D = %d" %(V,D))
  
    # Train model and get result -------------
    pmodel = pLSA()
    # 对topic个数为3 到 D+1 进行遍历，因为每次计算过程时间比较长，并且结果差的不错，所以在这里选取每隔10个进行计算
    for z in range(3,(D+1), 10):
      (l, p_d_z, p_w_z, p_z)  = pmodel.train(t_d, z, eps)
      logging.info('z = %d eps = %f' %(z, l))
      for itz in range(z):
        logging.info('Topic %d' %itz)
        data = [(p_w_z[i][itz], i)  for i in range(len(p_w_z[:,itz])) ]
        # 选择概率最大的 key_word_size 个数的word输出
        data.sort(key=lambda tup:tup[0], reverse=True)
        for i in range(key_word_size):
          logging.info('%s : %.6f ' %(pp.get_word(data[i][1]), data[i][0] ))

#### Run ####
`python main.py`

### Final average log likelihood and time cost of different topics K ###
对某个特定topic个数z，每次计算的时间比较长，所以在这里，只计算了`z in range(3,323, 10)` 范围内的数据。
如下：

z 表示topic的个数， llhood 表示log likelihood的值， cost表示计算所消耗的时间(s)

    z = 3 llhood = -973782.206921 cost = 747.666631
    z = 13 llhood = -922296.631344 cost = 953.463684
    z = 23 llhood = -894730.609518 cost = 1101.267654
    z = 33 llhood = -877589.227764 cost = 985.553553
    z = 43 llhood = -863563.797595 cost = 1024.922029
    z = 53 llhood = -852814.244978 cost = 1165.518772
    z = 63 llhood = -844012.122190 cost = 1141.052079
    z = 73 llhood = -836411.524585 cost = 1259.596428
    z = 83 llhood = -829528.584949 cost = 1341.569783
    z = 93 llhood = -823315.078888 cost = 1262.947609
    z = 103 llhood = -818415.021816 cost = 1512.228595
    z = 113 llhood = -813328.036424 cost = 1516.568643
    z = 123 llhood = -808700.478114 cost = 1663.149710
    z = 133 llhood = -804943.965458 cost = 1667.525403
    z = 143 llhood = -801118.183437 cost = 1751.649206
    z = 153 llhood = -796742.117443 cost = 1828.503664
    z = 163 llhood = -794702.480238 cost = 1858.348440
    z = 173 llhood = -790803.721784 cost = 2147.987550
    z = 183 llhood = -787822.730953 cost = 2038.396444
    z = 193 llhood = -784891.046263 cost = 1886.757959
    z = 203 llhood = -782920.879800 cost = 2199.012683
    z = 213 llhood = -780215.572185 cost = 2214.484940
    z = 223 llhood = -777977.077787 cost = 2264.572652
    z = 233 llhood = -776055.981161 cost = 2533.987723
    z = 243 llhood = -773836.837297 cost = 2415.084260
    z = 253 llhood = -772268.738924 cost = 2732.307158
    z = 263 llhood = -769585.005643 cost = 2530.018818
    z = 273 llhood = -767708.159138 cost = 2896.458289
    z = 283 llhood = -765369.237016 cost = 2754.486797
    z = 293 llhood = -763773.227083 cost = 3025.004390
    z = 303 llhood = -762708.746233 cost = 2787.175699
    z = 313 llhood = -761130.710307 cost = 3200.206291
    z = 323 llhood = -759153.088053 cost = 3018.660368

### Explain Topics in sides 12 ###
最后的数据见 result_verbose.dat

说明：
z : 表示topic个数为z时候的情况
cost :  表示计算所消耗的时间
llhod : log likelihood
topic : 一个数组，包含z个元素，每个元素是一个map，map的key是关键字，map的value是概率P(w|z)

如下数据是在z = 3时候的数据
    {3: {'cost': 747.666631,
         'llhood': -973782.206921,
         'topic': [[{'web': 0.045116},
                    {'information': 0.030074},
                    {'data': 0.029755},
                    {'semantic': 0.024606},
                    {'learning': 0.024484},
                    {'classification': 0.017352},
                    {'extraction': 0.012948},
                    {'planning': 0.012565},
                    {'text': 0.012329},
                    {'mining': 0.012174}],
                   [{'learning': 0.036771},
                    {'knowledge': 0.022713},
                    {'support': 0.014663},
                    {'models': 0.013992},
                    {'logic': 0.013513},
                    {'planning': 0.011569},
                    {'language': 0.011069},
                    {'vector': 0.010015},
                    {'probabilistic': 0.009392},
                    {'machine': 0.008913}],
                   [{'systems': 0.025266},
                    {'agents': 0.020213},
                    {'based': 0.017661},
                    {'model': 0.017511},
                    {'agent': 0.016911},
                    {'multi-agent': 0.012958},
                    {'framework': 0.011407},
                    {'analysis': 0.009565},
                    {'system': 0.008853},
                    {'design': 0.008751}]],
         'z': 3}}

