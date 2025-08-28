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
