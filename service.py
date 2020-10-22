'''
Descripttion: 
version: Python 3.6.3
Author: 王瑞国
Date: 2020-10-14 14:26:04
LastEditors: 王瑞国
LastEditTime: 2020-10-14 14:26:22
'''
import tornado
from tornado.web import Application
from tornado.httpserver import HTTPServer
from tornado.options import define
from tornado.options import options
from tornado.web import RequestHandler


# 为tornado构造的每个基类必须继承tornado.web中的RequestHandler对象
class HelloWord(RequestHandler):
    def get(self):
        self.write('Hello World')

def main():

    # 通过define方法定义属性及值，该属性会自动创建为options对象的属性
    define('port', default=10000, help='port to listen on')

    # Application 处理路由和试图的连接，以及运行tornado应用程序所需的设置
    application = Application([
        (r'/hello', HelloWord) # 完成接口地址为： http://ip:port/hello
    ])
    # 实例化tornado的HTTPServer，tornado用自己的服务器提供服务
    http_server = HTTPServer(application)

    # 指定监听端口，但是并不会启动服务器
    http_server.listen(options.port)
    print('Listention on ...')

    # 监听请求并返回对应的响应，tornado.ioloop.IOLoop开箱即用，以循环的形式监听端口
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()