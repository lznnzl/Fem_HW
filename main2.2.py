# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 12:10:26 2021

@author: Zn
"""

import numpy as np
import pandas as pd
import math as math
import pickle
import matplotlib.pyplot as plt
# from pyautocad import Autocad, APoint, aDouble



'尝试解决在计算节点施加力时计算不准确的问题'



class BeamSolve():
    '''
    作业3：采用梁单元，用于计算平面问题，计算几何非线性
    '''
    
    def __init__(self):
        
        # 构造函数
        # 原坐标，始终不变
        self.node0 = None
        self.node_num = None
        self.ele = None
        self.ele_num = None
        self.free_num = None
        
        # 截面信息
        self.sec_info = None
        
        # 定义累计变量
        self.node = None
        self.now_force = None

        # 记录变量
        self.rec_dis = None
  
        # 定义内部所需变量，初始为None，计算时再赋值
        self.vec_dis = None             # 位移向量
        self.vec_load = None            # 荷载向量
        self.free_info= None            # 单元释放信息，2列矩阵，       
        
        '节点位移，在全局坐标系'
        self.mat_dis = None

        # 全部材料的弹性模量
        self.E = 6000
        

        
    def read_3d3s(self,filename):
        
        def get_content(data,name):
            '''
            该函数用于获取部分段落内容。
            '''
            content = []
            
            i = data.index([name])
            
            while 1:
                i=i+1
                if data[i]!=['']:
                    content.append(data[i])
                else:
                    break
            return content
          
        
        # 自由函数，用于读取3d3s文件。
        fo=open(filename)
        data=[]
        node=[]
        ele=[]
        
        for line in fo:
            line =line.replace('\n','')
            if line=='*LAYERAXIS':
                break
            data.append(line.split(','))  
        fo.close()
        
        node = get_content(data,'*NODE')
        ele = get_content(data,'*ELE_LINE')
        sec = get_content(data,'*SECTION')

        node = np.array(node)[:,1:4]
        ele = np.array(ele)[:,2:7]
        

        # 数据处理及输出
        
        # 截面信息数据: 0截面类型，1-4截面参数, 6为杆件面积，7为强轴惯性矩
        '17为对称工字型截面，21为矩形管截面，29为矩形截面'
        sec_len = int(len(sec)/2)
        secdata = np.zeros((sec_len,8))
        
        
        '计算截面数据'
        for i in range(sec_len):
            secdata[i,0] = sec[int(2*i)][0]
            secdata[i,7] = sec[int(2*i)][1]
            
            secdata[i,1:5] = sec[int(2*i)][5:9]
            
            if secdata[i,0] == 21:
                '矩形管截面'
                h = secdata[i,1]
                b = secdata[i,2]
                tw = secdata[i,3]
                tf = secdata[i,4]
                h2=h-2*tf
                b2=b-2*tw
                
                Area = h*b-h2*b2
                I3 = 1/12*(b*h**3-b2*h2**3)  
                
                secdata[i,5] = Area
                secdata[i,6] = I3
                
            elif secdata[i,0] == 29:
                '矩形截面'
                h = secdata[i,1]
                b = secdata[i,2]
                Area = h * b
                I3 = 1/12 * ( h ** 3 * b)
                secdata[i,5] = Area
                secdata[i,6] = I3
            
            else:
                raise ValueError('截面类型溢出')
            

        # 节点数据: 节点坐标x, 节点坐标y, 节点坐标z
        nodedata = node.astype(np.float64)
        eledata = np.zeros((len(ele),4))
        
        
        # 杆件数据： 节点i，节点j，杆件类型号，杆件编号，
        eledata[:,0:2] = ele[:,3:5]
        eledata -= 1
        eledata[:,2:4] = ele[:,1:3]
        eledata = eledata.astype(np.int32)
        

        # 截面数据归并，将不同的截面数据归成同类
        ele_info_tp = np.zeros(len(eledata))
        for i in range(len(eledata)):    
            for j in range(len(secdata)):
                if eledata[i,2] == secdata[j,0] and eledata[i,3] == secdata[j,7]:
                    ele_info_tp[i] = j

        # 重新修改截面编号
        for i in range(len(eledata)):
            eledata[i,3] = ele_info_tp[i]
        
        
        # 参数输出
        #--------------------------------------------------------
        self.node = nodedata
        self.ele = eledata
        self.sec_info = secdata
        self.node_num = len(self.node)
        self.ele_num = len(self.ele)
        self.node0 = nodedata.copy()
        
        #--------------------------------------------------------
        self.free_num = 3 * self.node_num
        
        return None


    def loadSet(self,filename):
        '''
        该函数用于设置荷载，将各种形式的荷载转化整等效节点荷载

        Returns
        -------
        None.

        '''
        # 参数输入
        free_num = self.free_num
        
        # 计算过程
        loaddata = pd.read_excel(filename,sheet_name='节点荷载').values
        load_info = loaddata.astype(np.float)
        loadnum = load_info.shape[0]
        # print(load_info)
        
        
        vec_load = np.zeros(free_num)
        
        for i in range(loadnum):
            free_i = load_info[i,0]*3 + load_info[i,1]
            free_i = int(free_i)
            vec_load[free_i] = load_info[i,2]

        self.vec_load = vec_load
        
        return None
    
    
    
    def constraintSet(self,filename):
        '''
        该函数用于设置边界条件和判断是否还存在自由度

        Returns
        -------
        None.

        '''
        # 参数输入
        free_num = self.free_num
        temp = pd.read_excel(filename, sheet_name='边界条件').values
        con_info = temp[0:,:].astype(np.float32)
        
        vec_dis = np.zeros(free_num)
        vec_dis[:] = np.nan      
        
        for i in range(len(con_info)):
            node_i = con_info[i,0]
            temp_raw = 3 * node_i + con_info[i,1]
            temp_raw = int(temp_raw)
            vec_dis[temp_raw] = con_info[i,2]
            pass
        
        self.vec_dis = vec_dis

        return None
    
    
    def freedomSet(self,filename):
        # 参数输入
        free_num = self.free_num
        
        # 数据读取
        temp = pd.read_excel(filename, sheet_name='杆端释放').values
        free_info = temp[0:,:].astype(np.float32)
        
        a = free_info.shape[1]
        if a != 2:
            raise ValueError('杆端释放xlsx数据错误')
        
        # 参数输出
        self.free_info = free_info
        
        pass
    
    
    
    
    
    def __unitMatCal(self,node,mat_ele_force):
        '''
        该函数用于计算每个单元的单元刚度矩阵。 
        维度：单元数 x 6 x 6

        Returns
        -------
        None.
        
        '''
        
        # 参数输入
        node_num = self.node_num
        ele = self.ele
        ele_num = self.ele_num
        E = self.E
        sec_info = self.sec_info
        free_info = self.free_info
        now_force = self.now_force
        
        # 定义变量
        mat_unit = np.zeros((ele_num,6,6))
        mat_unit_geo = np.zeros((ele_num,6,6))

        for i in range(ele_num):
            # 生成弹性刚度矩阵
            # 截面数据
            sectype = ele[i,3]
            A = sec_info[sectype,5]
            I = sec_info[sectype,6]
            
            # 端点坐标数据
            node1 = node[ele[i,0]]
            node2 = node[ele[i,1]]
            vec = node2-node1
            l = np.linalg.norm(vec)

            # 两端刚接的弹性刚度矩阵
            k1 = E*A/l
            k2 = E*I/l/l/l
            k3 = E*I/l/l
            k4 = E*I/l
            
            mat_unit[i]=[
                [k1 ,   0,      0,      -k1,    0,          0],
                [0,     12*k2,  6*k3,   0,      -12*k2,     6*k3],
                [0,     6*k3,   4*k4,   0,      -6*k3,      2*k4],
                [-k1,   0,      0,      k1,     0,          0],
                [0,     -12*k2, -6*k3,  0,      12*k2,      -6*k3],
                [0,     6*k3,  2*k4,   0,      -6*k3,       4*k4]     
                ]
            
        # 对杆端释放的杆件的刚度矩阵进行修改    
        freeinfonum = free_info.shape[0]
        for i in range(freeinfonum):
            elei = int(free_info[i,0]) 
            
            sectype=ele[elei,3]
            # print(sectype)
            A=sec_info[sectype,5]
            I=sec_info[sectype,6]
            node1num=ele[elei,0]
            node2num=ele[elei,1]
            
            node1loc=[node[node1num,0],node[node1num,1]]
            node2loc=[node[node2num,0],node[node2num,1]]
            l=((node1loc[0]-node2loc[0])**2+(node1loc[1]-node2loc[1])**2)**0.5
            
            elenum = int(free_info[i,0])
            elenodenum = int(free_info[i,1])
            elenodenumi = ele[elenum,0]
            
            k1 = E*A/l
            k2 = E*I/l/l/l
            k3 = E*I/l/l
            k4 = E*I/l

            if elenodenum == -1:
                mat_unit[elei]=[
                    [k1,    0,      0,      -k1,    0,      0],
                    [0,     0,      0,      0,      0,      0],
                    [0,     0,      0,      0,      0,      0],
                    [-k1,   0,      0,      k1,     0,      0],
                    [0,     0,      0,      0,      0,      0],
                    [0,     0,      0,      0,      0,      0]     
                    ]
                
            elif elenodenum == elenodenumi:
                mat_unit[elei]=[
                    [k1,    0,          0,      -k1,    0,      0],
                    [0,     3*k2,       0,      0,      -3*k2,   3*k3],
                    [0,     0,          0,      0,      0,      0],
                    [-k1,   0,          0,      k1,     0,      0],
                    [0,     -3*k2,      0,      0,      3*k2,   -3*k3],
                    [0,     3*k3,       0,      0,      -3*k3,   3*k4]     
                    ]
                
            else:
                mat_unit[elei]=[
                    [k1,    0,      0,      -k1,    0,      0],
                    [0,     3*k2,   3*k3,   0,      -3*k2,  0],
                    [0,     3*k3,   3*k4,   0,      -3*k3,  0],
                    [-k1,   0,      0,      k1,     0,      0],
                    [0,     -3*k2,  -3*k3,  0,      3*k2,   0],
                    [0,     0,      0,      0,      0,      0]     
                    ]
        
                
        for i in range(ele_num):
            # 生成几何刚度矩阵
            Fx = mat_ele_force[i,3]
            Mi = mat_ele_force[i,2]
            Mj = mat_ele_force[i,5]
        
            e11 = Fx / l
            e13 = Mi / l
            e16 = Mj / l
            e22 = 12 * Fx * I / A / l**3 + 6 * Fx / 5 / l
            e23 = 6 * Fx * I / A / l**2 + Fx / 10
            e33 = 4 * Fx * I / A / l + 2 * Fx * l / 15
            e36 = 2 * Fx * I / A / l - Fx * l / 30
            
            mat_unit_geo[i] = [
                [e11,   0,      -e13,   -e11,   0,      -e16],
                [0,     e22,    e23,    0,      -e22,   e23],
                [-e13,  e23,    e33,    e13,    -e23,   e36],
                [-e11,  0,      e13,    e11,    0,      e16],
                [0,     -e22,   -e23,   0,      e22,    -e23],
                [-e16,  e23,    e36,    e16,    -e23,   e33]
                ]

        # 参数输出 弹性刚度矩阵加几何刚度矩阵
        mat_unit = mat_unit + mat_unit_geo

        return mat_unit


    def __cal_tran(self,node):
        '''
        该函数用于将单元刚度矩阵进行坐标变换

        Returns
        -------
        None.

        '''
         # 参数输入
        ele = self.ele
        ele_num = self.ele_num

        # 定义变量
        mat_tran = np.zeros((ele_num,6,6))
        
        for i in range(ele_num):
            node1 = node[ele[i,0]]
            node2 = node[ele[i,1]]
            vec = node2 - node1
            l = np.linalg.norm(vec)
            cc = vec[0] / l
            ss = vec[1] / l
            
            lamda = np.array([[cc,  ss, 0,  0,      0,  0],
                              [-ss, cc, 0,  0,      0,  0],
                              [0,   0,  1,  0,      0,  0],
                              [0,   0,  0,  cc,     ss, 0],
                              [0,   0,  0,  -ss,    cc, 0],
                              [0,   0,  0,  0,      0,  1]])
            
            mat_tran[i] = lamda
        
        return mat_tran


    def __tranMat(self,mat_unit,mat_tran):
        '''
        该函数用于将单元刚度矩阵进行坐标变换

        Returns
        -------
        None.

        '''
        
        # 参数输入
        ele_num = self.ele_num

        # 计算过程
        mat_unitTM = np.zeros((ele_num,6,6))
        for i in range(ele_num):            
            lamda = mat_tran[i]
            Ke0 = mat_unit[i]
            temp = np.dot(Ke0,lamda)
            Ke = np.dot(lamda.T,temp)
            mat_unitTM[i] = Ke
            
  
        return mat_unitTM


    def __totalMatCal(self,mat_unitTM):
        '''
        该函数用于将单元刚度矩阵组合成总刚矩阵

        Returns
        -------
        None.

        '''
        
        # 参数输入
        ele = self.ele
        ele_num = self.ele_num
        free_num = self.free_num

        # 定义矩阵
        mat_total = np.zeros((free_num,free_num))
        
        # 计算过程
        pos_data = np.zeros((ele_num,6))
        
        # 计算组装时位置数据
        for i in range(ele_num):
            pos_data[i,0] = 3 * ele[i,0]
            pos_data[i,1] = 3 * ele[i,0] + 1
            pos_data[i,2] = 3 * ele[i,0] + 2
            pos_data[i,3] = 3 * ele[i,1] 
            pos_data[i,4] = 3 * ele[i,1] + 1
            pos_data[i,5] = 3 * ele[i,1] + 2
        pos_data = pos_data.astype(np.int32)
        
        # 单元刚度矩阵组装成整体刚度矩阵
        for i in range(ele_num):
            for j in range(6):
                for k in range(6):
                    pos_i = pos_data[i,j]
                    pos_j = pos_data[i,k]
                    mat_total[pos_i,pos_j] += mat_unitTM[i,j,k]
        
        # 结果输出
        # self.mat_total = mat_total

        return mat_total
    
    
    def __equationSolve(self,mat_total,vec_load):
        
        '''
        该函数用于处理总刚矩阵及荷载向量，从而用于求解，引入边界条件

        Returns
        -------
        None.

        '''
        
        # 参数输入
        free_num = self.free_num
        node_num = self.node_num
        
        '边界条件的引入'
        vec_dis = self.vec_dis
        
        # 定义求解计算所需的矩阵
        mat_total_slove = np.copy(mat_total)
        vec_load_solve = np.copy(vec_load)
        
        
        # 大数放大倍数
        times = 1e10       
        # 对矩阵引入边界条件
        for i in range(free_num):
            if np.isnan(vec_dis[i]):
                pass
            
            elif vec_dis[i] == 0:
                mat_total_slove[i,:] = 0
                mat_total_slove[:,i] = 0
                mat_total_slove[i,i] = 1
                vec_load_solve[i] = 0
                
            elif vec_dis[i] != 0:
                mat_total_slove[i,i] *= times
                vec_load_solve[i] = mat_total_slove[i,i] * vec_dis[i]
                
        
        # 求解过程        
        vec_dis_result = np.linalg.solve(mat_total_slove,vec_load_solve)
        
        
        # 结果处理
        # 节点位移矩阵，整体坐标系
        mat_dis = np.zeros((node_num,3))
        for i in range(node_num):
            mat_dis[i,0] = vec_dis_result[3*i]
            mat_dis[i,1] = vec_dis_result[3*i+1]
            mat_dis[i,2] = vec_dis_result[3*i+2]
        
        return mat_dis,vec_dis_result   

    
    def __process_Vec(self,vec):
        vec_dis = self.vec_dis
        
        vec2 = vec.copy()
        
        for i in range(len(vec_dis)):
            if vec_dis[i] == 0:
                vec2[i] = 0 

        return vec2
    

    def cal_ela_dis(self,prv_node,mat_dis):
        # 参数输入
        ele = self.ele
        ele_num = self.ele_num
        
        
        # 计算过程     
        
        # 杆件两端位移矩阵，整体坐标系
        mat_ele_dis = np.zeros((ele_num,6))
        for i in range(ele_num):
            mat_ele_dis[i,0:3] = mat_dis[ele[i,0]]
            mat_ele_dis[i,3:6] = mat_dis[ele[i,1]]
        mat_tran = self.__cal_tran(prv_node)

        
        # 杆件位移，转换到局部坐标系
        mat_ele_disTM = np.zeros_like(mat_ele_dis)
        for i in range(ele_num):
            mat_ele_disTM[i] = np.dot(mat_tran[i],mat_ele_dis[i])
            


        '分离弹性变形'
        mat_dis_ela = np.zeros_like(mat_ele_disTM)
        for i in range(ele_num):
            node1 = prv_node[ele[i,0]]
            node2 = prv_node[ele[i,1]]
            l = np.linalg.norm(node2-node1)
            
            X12 = l + mat_ele_disTM[i,3] - mat_ele_disTM[i,0]
            Z12 = mat_ele_disTM[i,4] - mat_ele_disTM[i,1]
            
            

            
            theta_r = math.atan(Z12 / X12)
            
            '这里还是有问题'
            
            if X12 < 0:
                print(mat_ele_disTM[i,3],mat_ele_disTM[i,3])
                theta_r += math.pi
                print('sdafadf')
                raise ValueError('x')
            
            
            
                
                # print(l)
                # print(X12,Z12)
                
                # raise ValueError("atan error")            
                        
            
            
            
            
            l2 = math.sqrt(X12 ** 2 + Z12 ** 2)

            mat_dis_ela[i,2] = mat_ele_disTM[i,2] - theta_r
            mat_dis_ela[i,3] = l2 - l
            mat_dis_ela[i,5] = mat_ele_disTM[i,5] - theta_r        

        return mat_dis_ela
    
    
    
    def cal_force_add(self,mat_unit,mat_dis_ela):
        # 参数输入
        ele = self.ele
        ele_num = self.ele_num
        
        
        # 计算过程
        step_ele_force = np.zeros((ele_num,6))
        
        for i in range(ele_num):
            step_ele_force[i] = np.dot(mat_unit[i],mat_dis_ela[i])

        return step_ele_force
    
    
    def cal_node_force_add(self,step_ele_force,now_node):
        # 参数输入
        free_num = self.free_num
        node_num = self.node_num
        ele_num = self.ele_num
        ele = self.ele
        
        
        # 计算过程
        step_forceTM = np.zeros_like(step_ele_force)
        node_force = np.zeros((node_num,3))
        vec_force_add = np.zeros(free_num)
        
        '按变形后的坐标转换'
        mat_tran = self.__cal_tran(now_node)
        
        for i in range(ele_num):
            step_forceTM[i] = np.dot(mat_tran[i].T,step_ele_force[i])
        
        '换算到节点中'
        for i in range(ele_num):
            node_force[ele[i,0]] += step_forceTM[i,0:3]
            node_force[ele[i,1]] += step_forceTM[i,3:6]
        
        # print(node_force)
        
        '转成向量'
        for i in range(node_num):
            vec_force_add[3*i] = node_force[i,0]
            vec_force_add[3*i+1] = node_force[i,1]
            vec_force_add[3*i+2] = node_force[i,2]        

        return vec_force_add
    
    
    
    def elastic_Slove(self,):
        # 进行弹性求解
        
        # 参数输入
        node_num = self.node_num
        ele_num  = self.ele_num
        free_num = self.free_num
        vec_load = self.vec_load
        now_ele_force = np.zeros((ele_num,6)) # 累计的杆端力
        
        
        # 计算过程
        now_node = self.node0.copy()
        vec_load = self.vec_load
        
        mat_unit = self.__unitMatCal(now_node,now_ele_force)
        mat_tran = self.__cal_tran(now_node)
        mat_unitTM = self.__tranMat(mat_unit,mat_tran)
        mat_total = self.__totalMatCal(mat_unitTM)
        mat_dis,vec_dis = self.__equationSolve(mat_total,vec_load)
        
        # 参数输出
        self.mat_dis = mat_dis
        self.vec_dis_result = vec_dis
        
        
        
        return 0
    
    

    def nolinear_NR(self,times = 10, err = 0.01):
        # 参数输入
        node_num = self.node_num
        ele_num  = self.ele_num
        free_num = self.free_num
        vec_load = self.vec_load
        
        
        # 数据初始化
        prv_node = self.node0.copy()   # 当前步上一个节点坐标
        now_node = self.node0.copy()   # 当前步节点坐标
        now_mat_dis = np.zeros((node_num,3))        # 累计变形
        now_force = np.zeros(free_num)      # 当前荷载
        now_force_err = np.zeros(free_num)  # 当前荷载偏差
        now_tag_load = np.zeros(free_num)   # 当前荷载步目标荷载
        now_ele_force = np.zeros((ele_num,6)) # 累计的杆端力
        step_ele_force = np.zeros((ele_num,6)) # 当前步的杆端力
        
        
        # 开始进行计算
        times_load = 1000
        times_sub = 50
        step = 1 / times_load
        
        
        '用于记录变量'
        rec_time = 100
        rec_node = np.zeros((rec_time,node_num,3))
        rec_dis = np.zeros((rec_time,node_num,3))
        rec_step = times_load / rec_time
        
        # 第i个荷载步 
        for i in range(times_load):
            load_i = i+1
            
            # 计算当前荷载步的目标荷载
            now_tag_load = load_i * step * vec_load
            now_force_err = now_tag_load - now_force
            
            for j in range(50):
                '开始计算'
                mat_unit = self.__unitMatCal(now_node,now_ele_force)
                mat_tran = self.__cal_tran(now_node)
                mat_unitTM = self.__tranMat(mat_unit,mat_tran)
                mat_total = self.__totalMatCal(mat_unitTM)
                
                '计算得到整体坐标中的变形'
                mat_dis,vec_dis = self.__equationSolve(mat_total,now_force_err)
                
                '坐标数据更新'
                prv_node = now_node.copy()
                now_node[:,0:2] += mat_dis[:,0:2]
                now_mat_dis += mat_dis
                
                '计算分离当前步弹性变形'
                mat_dis_ela = self.cal_ela_dis(prv_node,mat_dis)
                
                
                '计算当前步弹性力增量，及杆端力'
                step_ele_force = self.cal_force_add(mat_unit,mat_dis_ela)
                now_ele_force += step_ele_force
                
                '计算当前节点力 并 更新当前力及力偏差'
                now_force = self.cal_node_force_add(now_ele_force,now_node)
                now_force_err = now_tag_load - now_force
                
                
                '判断是否满足收敛条件'
                now_force_err = self.__process_Vec(now_force_err)
                
                
                lg = np.linalg.norm(now_force_err)
                if lg < 1e-5:
                    print(i,j,lg)
                    # print(now_ele_force[9,1])
                    break
                pass
                
                if j == 49:
                    print(load_i * step)
                    raise ValueError('不收敛')
            
            
            if load_i % rec_step == 0:
                rec_i = int(load_i / rec_step -1)
                # 记录中间数值
                rec_node[rec_i] = now_node
                rec_dis[rec_i] = now_mat_dis
            
            
        # 结果输出
        # print(now_node)
        
        self.mat_dis = now_mat_dis
        self.node = now_node
        self.rec_node = rec_node
        self.rec_dis = rec_dis
        self.now_force = now_force
        self.now_ele_force = now_ele_force

        return 0
    
    
    def nolinear_Euler(self):
        '欧拉法，考虑平衡矫正'
        
        # 参数输入
        node_num = self.node_num
        ele_num  = self.ele_num
        free_num = self.free_num
        vec_load = self.vec_load
        
        
        # 数据初始化
        prv_node = self.node0.copy()   # 当前步上一个节点坐标
        now_node = self.node0.copy()   # 当前步节点坐标
        now_mat_dis = np.zeros((node_num,3))        # 累计变形
        now_force = np.zeros(free_num)      # 当前荷载
        now_force_err = np.zeros(free_num)  # 当前荷载偏差
        now_tag_load = np.zeros(free_num)   # 当前荷载步目标荷载
        now_ele_force = np.zeros((ele_num,6)) # 累计的杆端力
        step_ele_force = np.zeros((ele_num,6)) # 当前步的杆端力
        
        # 开始进行计算
        times_load = 1000
        step = 1 / times_load
        
        '用于记录变量'
        rec_time = 100
        rec_node = np.zeros((rec_time,node_num,3))
        rec_dis = np.zeros((rec_time,node_num,3))
        rec_step = times_load / rec_time
        
        # 开始进行计算
        for i in range(times_load):
            
            load_i = i+1
            
            # 计算当前荷载步的目标荷载
            now_tag_load = load_i * step * vec_load
            now_force_err = now_tag_load - now_force
            

            mat_unit = self.__unitMatCal(now_node,now_ele_force)
            mat_tran = self.__cal_tran(now_node)
            mat_unitTM = self.__tranMat(mat_unit,mat_tran)
            mat_total = self.__totalMatCal(mat_unitTM)
            
            '计算得到整体坐标中的变形'
            mat_dis,vec_dis = self.__equationSolve(mat_total,now_force_err)
            
            
            '坐标数据更新'
            prv_node = now_node.copy()
            now_node[:,0:2] += mat_dis[:,0:2]
            now_mat_dis += mat_dis
            
            '计算分离当前步弹性变形'
            mat_dis_ela = self.cal_ela_dis(prv_node,mat_dis)

            '计算当前步弹性力增量'
            step_ele_force = self.cal_force_add(mat_unit,mat_dis_ela)
            now_ele_force += step_ele_force
            
            '更新当前力及力偏差'
            now_force = self.cal_node_force_add(now_ele_force,now_node)

            '记录中间数据'
            if load_i % rec_step == 0:
                rec_i = int(load_i / rec_step -1)
                # 记录中间数值
                rec_node[rec_i] = now_node
                rec_dis[rec_i] = now_mat_dis
            pass
            
            print(i)
        
        # 参数输出
        self.rec_node = rec_node
        self.node = now_node
        
        pass
    

    def nolinear_Arc_len(self, method = 0, loop_times = 500, scale = 5):
        
        
        '采用弧长法进行计算'
        '这里先采用超平面约束 考虑超平面的更新'
        
        # 输出分析类型
        if method == 0:
            print('当前采用超平面约束')
        elif method == 1:
            print('当前采用更新的超平面约束')
        elif method == 2:
            print('当前采用的超球面约束')


        # 参数输入
        node_num = self.node_num
        ele_num  = self.ele_num
        free_num = self.free_num
        vec_load = self.vec_load
        
        # 数据初始化
        prv_node = self.node0.copy()            # 当前步上一个节点坐标
        now_node = self.node0.copy()            # 当前步节点坐标
        now_mat_dis = np.zeros((node_num,3))    # 累计变形
        now_force = np.zeros(free_num)          # 当前荷载
        now_force_err = np.zeros(free_num)      # 当前荷载偏差
        now_tag_load = np.zeros(free_num)       # 当前荷载步目标荷载
        now_ele_force = np.zeros((ele_num,6))   # 累计的杆端力
        step_ele_force = np.zeros((ele_num,6))  # 当前步的杆端力
        now_lamda = 0   # 累计的荷载因子        
        
        step_vec_dis = 0
        step_lamda = 0
        
        
        # 开始进行计算
        '约束弧长'
        times_load = loop_times
        scale = scale
        
        
        
        # 进行一次线性求解，计算近似弧长
        self.elastic_Slove()
        x = self.vec_dis_result
        r0 = np.linalg.norm(x) / times_load * scale
        
        end_time = 1.0
                
        '记录变量'
        rec_dis = np.zeros((times_load+1, 2))

        
        # 开始计算
        for i in range(times_load):
            
            # 该荷载步的第一次计算

            '开始计算'
            mat_unit = self.__unitMatCal(now_node,now_ele_force)
            mat_tran = self.__cal_tran(now_node)
            mat_unitTM = self.__tranMat(mat_unit,mat_tran)
            mat_total_0 = self.__totalMatCal(mat_unitTM)
            
            '计算得到整体坐标中的变形'
            xxx,vec_dis_0 = self.__equationSolve(mat_total_0,vec_load)
            lamda = r0 / math.sqrt(np.dot(vec_dis_0,vec_dis_0)+1)
            
            '判断lamda正负'
            if i == 0:
                pass
            
            else:
                last_r = np.r_[step_vec_dis,step_lamda]
                now_r = np.r_[vec_dis_0,1]
                
                tag = np.dot(last_r,now_r)
                
                if tag <0:
                    lamda = -lamda
                    print(i,'下降段')
            
            
            
            now_lamda += lamda
            step_lamda = lamda
            step_vec_dis = None
            
            if now_lamda > end_time:
                now_lamda = end_time
            
            
            '设置第一步计算的目标力'
            now_tag_load = now_lamda * vec_load
            now_force_err = now_tag_load - now_force
            
            
            # 第一步求解
            '进行第一步的位移求解'
            mat_dis,vec_dis = self.__equationSolve(mat_total_0,now_force_err)

            # now_r = np.r_[vec_dis,lamda]
            
            '坐标数据更新'
            prv_node = now_node.copy()
            now_node[:,0:2] += mat_dis[:,0:2]
            now_mat_dis += mat_dis
            step_vec_dis = vec_dis
            
            '计算分离当前步弹性变形'
            mat_dis_ela = self.cal_ela_dis(prv_node,mat_dis)
            
            '计算当前步弹性力增量'
            step_ele_force = self.cal_force_add(mat_unit,mat_dis_ela)
            now_ele_force += step_ele_force

            '更新当前力及力偏差'
            now_force = self.cal_node_force_add(now_ele_force,now_node)
            now_force_err = now_tag_load - now_force
            
            
            
            # 该荷载步的后续计算
            for j in range(30):
                
                # 计算该步的 delta_lamda
                mat_unit = self.__unitMatCal(now_node,now_ele_force)
                mat_tran = self.__cal_tran(now_node)
                mat_unitTM = self.__tranMat(mat_unit,mat_tran)
                mat_total = self.__totalMatCal(mat_unitTM)
                xxx,vec_dis1 = self.__equationSolve(mat_total,now_force_err)
                xxx,vec_dis2 = self.__equationSolve(mat_total,vec_load)
                
                if method == 0:
                    # 超平面约束
                    delta_lamda = - np.dot(vec_dis_0,vec_dis1)/(np.dot(vec_dis_0,vec_dis2)+1)
                elif method == 1:
                    # 超平面更新
                    delta_lamda = (- np.dot(step_vec_dis,vec_dis1)
                                   /(np.dot(step_vec_dis,vec_dis2)+step_lamda))
                elif method == 2:
                    # 超球面约束
                    a = np.dot(vec_dis2,vec_dis2)+1
                    b = 2 * (np.dot(vec_dis2,(vec_dis1 + step_vec_dis)) + step_lamda)
                    c = np.dot(vec_dis1,(vec_dis1 + 2 * step_vec_dis))
                    
                    delta = b ** 2 - 4 * a * c
                    
                    if delta < 0:
                        raise ValueError('can not cal ')
                    
                    delta = math.sqrt(delta)
                    
                    x1 = (-b + delta)/(2 * a)
                    x2 = (-b - delta)/(2 * a)
                    
                    # print(x1,x2)
                    '如何判断选取哪个呢'
                    
                    delta_lamda = x1
                else:
                    raise ValueError('method error')
                    
                    
                
                # 计算该步的变形
                
                '首先更新目标力'
                now_lamda += delta_lamda
                step_lamda += delta_lamda
                
                if now_lamda > end_time:
                    now_lamda = end_time
                
                
                
                now_tag_load = now_lamda * vec_load
                now_force_err = now_tag_load - now_force
                
                '该步求解'
                mat_dis,vec_dis = self.__equationSolve(mat_total,now_force_err)

                '坐标数据更新'
                prv_node = now_node.copy()
                now_node[:,0:2] += mat_dis[:,0:2]
                now_mat_dis += mat_dis
                step_vec_dis += vec_dis
                
                '计算分离当前步弹性变形'
                mat_dis_ela = self.cal_ela_dis(prv_node,mat_dis)
    
                '计算当前步弹性力增量'
                step_ele_force = self.cal_force_add(mat_unit,mat_dis_ela)
                now_ele_force += step_ele_force

                '更新当前力及力偏差'
                now_force = self.cal_node_force_add(now_ele_force,now_node)
                now_force_err = now_tag_load - now_force
                
                
                # 判断是否收敛
                '将支座处的内力归零'
                now_force_err = self.__process_Vec(now_force_err)
                
                lg = np.linalg.norm(now_force_err)
                if lg < 1e-3:
                    if i % 10 == 0:
                        print(i,j,lg)
                        # print(np.linalg.norm(now_r))
                    break
                
            if now_lamda >= 1:
                break
                        
            rec_dis[i+1,0] = now_lamda
            rec_dis[i+1,1] = -now_mat_dis[3,1]
        
        rec_dis = rec_dis[0:i+2,:]
        
        
        # 结果输出
        print(now_lamda)
        
        self.node = now_node
        self.rec_dis = rec_dis
        

    def plot_structure(self,node = None):
        # 参数输入
        ele = self.ele
        ele_num = len(ele)
        
        
        if node is None:
            node = self.node0
        
        node = self.node0
        
        # 绘制过程
        for i in range(ele_num):
            node1 = node[ele[i,0]]
            node2 = node[ele[i,1]]
            x = [node1[0],node2[0]]
            y = [node1[1],node2[1]]
            plt.plot(x,y,'black')
            pass
        plt.axis('equal')
        plt.show()
        pass


    def divide_Ele(self,divide_num = 3):
        # 将杆件等分
        
        # 参数输入
        node = self.node0
        ele = self.ele
        node_num = self.node_num
        ele_num = self.ele_num
        
        
        '定义新的节点及杆件数据'
        new_node_num = node_num + (divide_num-1) * ele_num
        new_node = np.zeros((new_node_num,3))
        
        new_ele_num = divide_num * ele_num
        new_ele = np.zeros((new_ele_num,4))
        
        new_node[0:node_num] = node
        
        
        # 计算过程
        '对每根杆件进行循环'
        for i in range(ele_num):
            node1 = node[ele[i,0]]
            node2 = node[ele[i,1]]
            
            '将一段划分成多段'
            node_i = int(node_num + (divide_num-1) * i)
            ele_i = int(divide_num * i)
            
            for j in range(divide_num):
                if j == 0:
                    new_ele[ele_i] = [ele[i,0], node_i, ele[i,2], ele[i,3]]
                elif j < divide_num -1:
                    new_ele[ele_i] = [node_i+j-1, node_i+j, ele[i,2], ele[i,3]]
                else:
                    new_ele[ele_i] = [node_i+j-1, ele[i,1], ele[i,2], ele[i,3]]    
                ele_i += 1
            
                '添加节点'
                if j != 0:                    
                    pt = ((divide_num - j) * node1 + j * node2) / divide_num
                    new_node[node_i + j - 1] = pt                    

        # 参数输出
        self.node = new_node
        self.ele = new_ele.astype(np.int32)
        self.node_num = len(self.node)
        self.ele_num = len(self.ele)
        self.node0 = self.node.copy()
        self.free_num = 3 * self.node_num



class end():
    pass

     
if __name__ ==  "__main__":
    
    
    # 前处理，输入模型相关数据
    beam = BeamSolve()
    
    # beam.read_3d3s('1.3D3S')
    # filename = 'dataSet.xlsx'
    
    
    beam.read_3d3s('2.3D3S')
    filename = 'dataSet2.xlsx'
    
    # beam.divide_Ele()
    
    '拱-算例'
    # beam.read_3d3s('arc.3D3S')
    # filename = 'dataSet_arc.xlsx'
    
    
    
    beam.constraintSet(filename)
    beam.freedomSet(filename)
    beam.loadSet(filename)
    
    
    
    
    
    
    
    # 弹性求解
    beam.elastic_Slove()

    
    

    # # 非线性求解 NR
    # beam.nolinear_NR()
    # xxx = beam.rec_node[:,10,1]

    

    # # Euler法
    # beam.nolinear_Euler()
    # xxx = beam.rec_node[:,10,1]
    
    
    # # 弧长法
    # beam.nolinear_Arc_len(0,500,10)
    

    # # 绘图
    # rec_dis = beam.rec_dis
    # plt.plot(rec_dis[:,1],rec_dis[:,0])
    
    
    

    # beam.plot_structure()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    