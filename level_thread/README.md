
## openblas multithreading frameworks; pull the api sgemm from openblas code
openblas看似很大其实只是因为需要实现众多API, 而绝大多数情况下我们只需要用sgemm这一个API,

因此就做了这个尝试,从openblas中抽出的sgemm实例,并且保留了openblas的多线程和内存优化的部分,

最终只需几个文件可以达到一样的效果，实现同样的功能，效率还略高些；

![image](https://github.com/wangshankun/wsk_lab/blob/master/level_thread/readme.jpg)


      
  
   
