from flask_bootstrap import Bootstrap
import json
from flask import Flask, request, render_template
from auto_abstract_impl import AutoAbstractImpl
import time

app = Flask(__name__)
bootstrap = Bootstrap(app)
instance_abstract = AutoAbstractImpl()
instance_abstract.load_mode()


@app.route('/')
def user():
    return render_template('my_form.html')


def calculate_function_run_time_ms(func):
    def call_fun(*args, **kwargs):
        start_time = time.time()
        f = func(*args, **kwargs)
        end_time = time.time()
        print('%s() run time：%s ms' % (func.__name__, int(1000 * (end_time - start_time))))
        return f

    return call_fun


# 用于测试
@app.route('/get_abstract/', methods=["GET", "POST"])
@calculate_function_run_time_ms
def auto_abstract():
    title = request.form['title']
    content = request.form['content']
    chi = request.form['chi']
    eng = request.form['eng']
    result = instance_abstract.get_abstract(title, content, chi, eng)
    return json.dumps(result, ensure_ascii=False)


if __name__ == '__main__':
    app.run('127.0.0.1', port=8001)  # 测试
    # app.run('0.0.0.0')  # 使用
