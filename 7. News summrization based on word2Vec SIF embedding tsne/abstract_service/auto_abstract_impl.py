# -*- coding: utf-8 -*-
"""
# @Time    : 2019/11/23 13:59
# @Author  : zhaobinghao
# @File    : auto_abstract_impl.py
"""
from auto_abstract import AutoAbstract
import json

instance_abstract = AutoAbstract()
result = {'status': 'failure',
          'msg': 'error',
          'data': ''}


class AutoAbstractImpl(object):
    @staticmethod
    def load_mode():
        global result
        try:
            instance_abstract.load_model()
            result = {'status': 'success',
                      'msg': '模型加载成功',
                      }
        except OSError as e:
            result = {'status': 'failure',
                      'msg': '模型加载路径错误',
                      }
        except Exception as e:
            result = {'status': 'failure',
                      'msg': '模型加载失败',
                      }
        return result

    @staticmethod
    def get_abstract(title, content, chi, eng):
        global result
        try:
            if chi == eng:
                result = {'status': 'failure',
                          'msg': '请选择一种文本摘要方式',
                          'data': ''}
                return result
            if not title or not content:
                result = {'status': 'failure',
                          'msg': '标题或正文不能为空',
                          'data': ''}
                return result
            pct_keep = 0.3 if len(content) >= 300 else 0.5
            if chi == 'true':
                abstract = instance_abstract.get_abstract(title, content,
                                                          seperator=r'。|\，|\！|\……|\（|\）|\？|\.|\,|\!|\?|\(|\)',
                                                          n_neigbors=3, pct_keep=pct_keep, weightpara=1e-3,
                                                          language='Chinese')
            else:
                abstract = instance_abstract.get_abstract(title, content,
                                                          seperator=r'。|\，|\！|\……|\（|\）|\？|\.|\,|\!|\?|\(|\)',
                                                          n_neigbors=3, pct_keep=pct_keep, weightpara=1e-3,
                                                          language='English')

            result = {'status': 'success',
                      'msg': '模型预测成功',
                      'data': abstract}
            # raise ZeroDivisionError  # test
        except Exception as e:
            print(e)
            result = {'status': 'failure',
                      'msg': '模型预测失败',
                      'data': ''}
            # raise e
        return result
