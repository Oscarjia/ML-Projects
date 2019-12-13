#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#获取每一行content列的内容，并单独输出到一个txt文件中
for i in range(df.shape[0]):
    fin = open("/Users/li/Desktop/AI/res_{}.txt".format(i), 'w')
    tmp = df.loc[i, 'content']
    if not isinstance(tmp, str) or not tmp.strip(): continue
    fin.write(tmp)
    fin.close()
#fin.close()


# In[106]:


#### 正则表达式处理文本
import re
#正则对字符串的清洗
def textParse(str_doc):
    #正则过滤掉特殊符号，标点，英文，数字等
    r1 = '[a-zA-Z0-9’！"#$%&\'()*+,-./:：;：｜<=>?@, -。?★、]^_`{|}~]+'
    #去除空格
    r2 = '\s+'
    str_doc = re.sub(r1,' ',str_doc)
    str_doc = re.sub(r2,' ',str_doc)
    #去除换行符
    str_doc = str_doc.replace('\n','')
    #去除“\n”字符
    str_doc = str_doc.replace('\\n','')
    #去除“↑”字符
    str_doc = str_doc.replace('↑','')
    str_doc = str_doc.replace('?','')
    str_doc = str_doc.replace('□','')
    str_doc = str_doc.replace('③','')
    return str_doc
    
def readFile(path):
	str_doc = ""
	with open(path,'r',encoding='utf-8') as f:
		str_doc = f.read()
	return str_doc

if __name__ == '__main__':
	# 1.读取文本
	path = r'/Users/li/Desktop/AI/RES3/res_6509.txt'
	str_doc = readFile(path)
	#print(str_doc)
    
    # 2.数据清洗
res = textParse(str_doc)
print(res)


# In[ ]:


#处理全部
import os
 
base_path = '/Users/li/Desktop/AI'  
 
for root, dirs, files in os.walk(base_path):  
    for file in files:
        if file[-3:] == 'txt':
            file_path = os.path.join(root, file)
            os.system("python3 deepzoom_tile -Q 0 -s 1024 name.txt")  
 
        else:
            continue
 
print('all done')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




