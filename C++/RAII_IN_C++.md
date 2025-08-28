# Agenda
- What is a resource?
- What issues do we have handling resources
- RAII to manage resources
- Examples in Standard
- Implementing a RAII class

## Resource
A resouurce in C++ is some facility or concept that you gain access to by a statement or expression, and you release or dispose of that facility or concept by some other statement or expression.

### Common Resources

| Resource | Acquire | Dispose |
|----------|----------|----------|
| Memory    | p = new T;     | delete p;     |
| POSIX File   | fp = fopen("filename", "r");     | fclose(fp);     |
| Joinable threads    | pthread_create(&p, NULL, fn, NULL);     | pthread_join(p, &retVal);     |
| Mutex locking    | pthread_mutex_lock(&mut);     | pthread_mutex_unlock(&mut);     |


### Resource Usage Issues
- Leak
- Use-after-disposal
- Double-disposal

We will use a mutex as the example resource.

#### A Sample Function

```c++
bool fn(std::mutex & someMutex, SomeDataSource & src)
{
    someMutex.lock();            // Locked the mutex here
    BufferClass buffer;
    src.readIntoBuffer(buffer);
    buffer.display();
    // someMutex.unlock           // If there is no unlock, add one
    return true;                  // Didn't unlocked the mutex
}
```
But as we exlpore more of this code, we find that there was no condition for returning false. Now, we need to add that , and then make a seperate unlock call. 
Now, what happens if there an exception? You use try and catch and unlock.
However, you will find a continuous problem with managing the memory which is related directly to the **lifetime of the object**.

### C++ Object Lifetimes
Objects have a defined beginning of life, and end of life. Both of those events have code which will automatically run: namely, constructors and destructors.

## RAII

** Resource Acquisition Is Initialization**
In the purest sense of term, this is the idom where resource acquisition is done in the constructor of an "RAII Resource", and resource disposal is done in the destructor of an "RAII class".

### Ownership
An RAII class is said to "own" the resource. It is responsible for cleaning up that resource at the appropiate time.

### RAII Example: std::lock_guard
`std::lock_guard` is the standard RAII class to lock a single mutex during its construction, and unlock it during desctruction. 
> Starting from ->

<img width="662" height="370" alt="image" src="https://github.com/user-attachments/assets/394b6197-422f-4361-8bdd-c15000852afe" />

> Now, we have ->

<img width="661" height="372" alt="image" src="https://github.com/user-attachments/assets/1d40ecae-66f3-4b45-8ebd-6764c5c1cf6e" />

Thus, having a RAII class simplified our API calls for resources. Now, we won't leak that mutex and it will unlock on it's own at the end of the scope.

### Storage Durations
So far we've only talked about automatic storage duration variables. RAII works with any of the C++ object lifetimes.

We can have this collection of worker threads with `std::jthread` , so that once they are used, they will go out of scope and complete their lifetime, thus giving back the ownership when all the threads are done.
```c++
void SomeClass::fn()
{
    auto worker {std::jthread { [] { /* do somnething */ }}};
    m_vec.push_back(std::move(worker));
}
```

#### RAII Example -> std::unique_ptr
`std::unique_ptr` is the standard RAII class to hold a pointer, and dispose of it during destruction. By default, this corresponds to `new` and `delete`.

#### RAII Example -> std::shared_ptr
`std::shared_ptr` represents a refrence-counted shared pointer.

#### Other Standard RAII classes
Some other Standard RAII classes:
- `std::unique_lock`: a more sophisticated `std::lock_guard`, but you can unlock and relock it during its lifespan, and other more sophisticated things.
- `std::jthread`: manages a joinable thread, and will automatically join to the thread during destruction.
- `std::fstream`: opens and closes the files.

### Reclamin Responsibility:
RAII classes may provide ways to get direct access to the enclosed resource.
RAII classes may even provide ways to break the resources out of the RAII class altogther.

### Solves the problems?
- Leaks?
      - Yes. Automatic storage durations, can't forget to dispose.
- Use-after-disposal?
      - Yes. After use , the variable goes out of scope.
- Double-disposal?
      - Yes. Local variables don't go out of space twice.
### NOT A GOD METHOD
RAII does not solves everything. For example, if two resources wants to talk to each other, RAII doesn't solve any problems related to that.
- Resource loops -> For example, a shared pointer in a doubly linked list will have a problem if one of the node is deleted.
- Deadloops -> For example, let's say if for one thread we take mutex A and Mutex B and for another thread, vice versa. What if the timing of asking for using this two mutex overlaps each other? It will create a deadlock. (we do have `std::scoped_lock` to fix this in a hindsight).

## Implementing a RAII class
 If the resource that youa re trying to wrap is represented as a pointer already, then you do not have to implement your own RAII class. `std::unique_ptr` is likely already able to manage the pointer for you.

Let's assume we still want to manage our file class:
### Custom Disposal method:
Let's assume we want to manage a normal FILE.
```c++
FILE * fopen (const char * filename, const char * mode);
int fclose(FILE * stream);
```
assuming above example, Now i am going to present a little boilerplate to make using this a little easier. First, a function which will be used to dispose of the FILE handle:

```c++
struct file_closer
{
    void operator() (FILE * stream) const { fclose(stream); }
};
```

Then, using a declaration to make my new RAII type:
```c++
using cfile = std::unique_ptr<FILE, file_closer>;
```
