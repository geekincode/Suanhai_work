## 一、编译流程

### 1.2 编译的四个阶段

| 阶段 | 主要作用 | 常见产物 |
|---|---|---|
| 预处理 | 展开 `#include`、处理宏定义等预处理指令 | `.i` |
| 编译 | 检查 C++ 语法并生成汇编代码 | `.s` |
| 汇编 | 将汇编代码转换为目标文件 | `.o` |
| 链接 | 将目标文件与库组合成可执行文件 | 可执行文件 |

``` bash 
g++ -E Hello.cpp -o Hello.i
g++ -S Hello.i -o Hello.s
g++ -c Hello.s -o Hello.o
g++ Hello.o -o Hello
./Hello
```