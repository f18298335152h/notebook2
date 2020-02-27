1. explicit作用：explicit TaskFaceDetectionNPU(const std::shared_ptr<sonic::ModelConfig> & model_config); 
（关键字只能用于类内部的构造函数声明上。）用来修饰构造函数，禁止隐士类型转换，只能显示的方式进行类型转换。
例子：
class things
{
    public:
        things(const std::string&name =""):
              m_name(name),height(0),weight(10){}
        int CompareTo(const things & other);
        std::string m_name;
        int height;
        int weight;
};
things a;
................//在这里被初始化并使用。
std::string nm ="book_1";
//由于可以隐式转换，所以可以下面这样使用
int result = a.CompareTo(nm);
　　这段程序使用一个string类型对象作为实参传给things的CompareTo函数。这个函数本来是需要一个tings对象作为实参。
　　现在编译器使用string nm来构造并初始化一个 things对象，新生成的临时的things对象被传递给CompareTo函数，并在离开这段函数后被析构。
 
 2. 构造函数使用delete TaskFaceDetectionNPU() = delete， default;
    
    MyClass()=default;  //同时提供默认版本和带参版本，类型是POD的
    myClass()=delete;//表示删除默认构造函数
    
    
    
C++ Traits
C++ 并不总是把 class 和 typename 视为等价。有时候我们一定得使用 typename。 
默认情况下，C++ 语言假定通过作用域运算符访问的名字不是类型。因此，如果我们希望使用一个模板类型参数的类型成员，就必须显式告诉编译器该名字是一个类型。我们通过使用关键字 typename 来实现这一点：

template<typename T>
typename T::value_type top(const T &c)
{
    if (!c.empty())
        return c.back();
    else
        return typename T::value_type();
}
使用typename使得模板在编译的时候知道， value_type是模板T的一个类型。


template和class定义模板时的区别：
由于 C++ 允许在类内定义类型别名，且其使用方法与通过类型名访问类成员的方法相同。故而，在类定义不可知的时候，编译器无法知晓类似 Type::foo 的写法具体指的是一个类型还是类内成员。
例如在以下代码中，类模板 Bar 的原意是使用类 Foo 实例化，而后引用其中的 bar_type 定义名为 bar 的类内成员。然而，就 T::bar_type 而言，编译器在编译期无法确定它究竟是不是一个类型。
此时就需要 typename 关键字来辅助编译器的判断。



const并未区分出编译期常量和运行期常量
constexpr限定在了编译期常量
constexpr：告诉编译器我可以是编译期间可知的，尽情的优化我吧。
const：告诉程序员没人动得了我，放心的把我传出去；或者放心的把变量交给我，我啥也不动就瞅瞅。



using的三种用法

1. 命名空间的使用using namespace android;
2. 在子类中引用基类的成员
    
class T5Base {
public:
    T5Base() :value(55) {}
    virtual ~T5Base() {}
    void test1() { cout << "T5Base test1..." << endl; }
protected:
    int value;
};
 
class T5Derived : private T5Base {
public:
    using T5Base::test1;
    using T5Base::value;
    void test2() { cout << "value is " << value << endl; }
};
基类中成员变量value是protected，在private继承之后，对于外界这个值为private，也就是说T5Derived的对象无法使用这个value。
如果想要通过对象使用，需要在public下通过using T5Base::value来引用，这样T5Derived的对象就可以直接使用。
同样的，对于基类中的成员函数test1()，在private继承后变为private，T5Derived的对象同样无法访问，通过using T5Base::test1 就可以使用了。


3. 别名指定
    using value_type = _Ty 以后使用value_type value; 就代表_Ty value
这个让我们想起了typedef，using 跟typedef有什么区别呢？哪个更好用些呢？
typedef void (*FP) (int, const std::string&);
若不是特别熟悉函数指针与typedef的童鞋，我相信第一眼还是很难指出FP其实是一个别名，代表着的是一个函数指针，而指向的这个函数返回类型是void，
接受参数是int, const std::string&。那么，让我们换做C++11的写法：
using FP = void (*) (int, const std::string&);
我想，即使第一次读到这样代码，并且知道C++11 using的童鞋也能很容易知道FP是一个别名，using的写法把别名的名字强制分离到了左边，而把别名指向的放在了右边，比较清晰。


关于强制类型转换的问题，很多书都讨论过，写的最详细的是C++ 之父的《C++ 的设计和演化》。最好的解决方法就是不要使用C风格的强制类型转换，
而是使用标准C++的类型转换符：static_cast, dynamic_cast。标准C++中有四个类型转换符：static_cast、dynamic_cast、reinterpret_cast、和const_cast
1. static_cast < type-id > ( expression )该运算符把expression转换为type-id类型，但没有运行时类型检查来保证转换的安全性
2. dynamic_cast < type-id > ( expression ) 该运算符把expression转换成type-id类型的对象。Type-id必须是类的指针、类的引用或者void *；
    如果type-id是类指针类型，那么expression也必须是一个指针，如果type-id是一个引用，那么expression也必须是一个引用。dynamic_cast主要用于类层次间的上行转换和下行转换，还可以用于类之间的交叉转换
    dynamic_cast具有类型检查的功能，比static_cast更安全。
3. ：const_cast<type_id> (expression) 该运算符用来修改类型的const或volatile属性。除了const 或volatile修饰之外， type_id和expression的类型是一样的
    常量指针被转化成非常量指针，并且仍然指向原来的对象；常量引用被转换成非常量引用，并且仍然指向原来的对象；常量对象被转换成非常量对象
    
mutable：
  mutable 的出现，将 C++ 中的 const 的概念分成了两种
  1. 二进制层面的 const，也就是「绝对的」常量，在任何情况下都不可修改（除非用 const_cast）
  2. 引入 mutable 之后，C++ 可以有逻辑层面的 const，也就是对一个常量实例来说，从外部观察，它是常量而不可修改；但是内部可以有非常量的状态。
  mutable 只能用来修饰类的数据成员；而被 mutable 修饰的数据成员，可以在 const 成员函数中修改。
  
 C++中对共享数据的存取在并发条件下可能会引起data race的undifined行为，需要限制并发程序以某种特定的顺序执行，
 有两种方式：使用mutex保护共享数据，原子操作：针对原子类型操作要不一步完成，要么不做，不可能出现操作一半被切换CPU，
 这样防止由于多线程指令交叉执行带来的可能错误。非原子操作下，某个线程可能看见的是一个其它线程操作未完成的数据。
 atomic<T>模板类，生成一个T类型的原子对象，并提供了系列原子操作函数。其中T是trivially  copyable type满足：要么全部定义了拷贝/移动/赋值函数，要么全部没定义;
 没有虚成员;基类或其它任何非static成员都是trivally copyable。典型的内置类型bool、int等属于trivally copyable。再如class triviall{public: int x};也是。
 T能够被memcpy、memcmp函数使用，从而支持compare/exchange系列函数。有一条规则：不要在保护数据中通过用户自定义类型T通过参数指针或引用使得共享数据超出保护的作用域。
 atomic<T>编译器通常会使用一个内部锁保护，而如果用户自定义类型T通过参数指针或引用可能产生死锁。总之限制T可以更利于原子指令
--------------------- 


在引入右值引用，转移构造函数，转移复制运算符之前，通常使用push_back()向容器中加入一个右值元素（临时对象）的时候，首先会调用构造函数构造这个临时对象，
然后需要调用拷贝构造函数将这个临时对象放入容器中。原来的临时变量释放。这样造成的问题是临时变量申请的资源就浪费。 
引入了右值引用，转移构造函数（请看这里）后，push_back()右值时就会调用构造函数和转移构造函数。 在这上面有进一步优化的空间就是使用emplace_back

emplace_back  类比push_back,都是往vector容器添加元素的操作:
    在容器尾部添加一个元素，这个元素原地构造，不需要触发拷贝构造和转移构造。而且调用形式更加简洁，直接根据参数初始化临时对象的成员。 

将拷贝构造和转移构造看在：复制和剪切的功能，剪切总比复制快，并且利用资源更充分
例子：
如果类President 声明了转移构造：push_back在插入一个对象时，不在调用拷贝构造，而是调用转移构造
std::vector<President> elections;
elections.emplace_back("Nelson Mandela", "South Africa", 1994); //没有类的创建
reElections.push_back(President("Franklin Delano Roosevelt", "the USA", 1936)); 创建临时对象，调用构造和转移构造

转移构造函数：
MyString(MyString&& str) { 
    std::cout << "Move Constructor is called! source: " << str._data << std::endl; 
    _len = str._len; 
    _data = str._data; 
    str._len = 0; 
    str._data = NULL; 
 }；

转移赋值操作符： 
MyString& operator=(MyString&& str) { 
    std::cout << "Move Assignment is called! source: " << str._data << std::endl; 
    if (this != &str) { 
      _len = str._len; 
      _data = str._data; 
      str._len = 0; 
      str._data = NULL; 
    } 
    return *this; 
 }
--------------------- 


