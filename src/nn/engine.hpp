#include <functional>
#include <string>
#include <vector>

namespace forgeml {
// Stores scalar values and it's gradient
class Value {
    double _data;
    double _grad = 0.0;
    std::vector<Value*> _prev;
    std::function<void()> _backward_fn;
    std::string _op;

    Value(double d) : _data(d) {}
    Value(double d, std::vector<Value*> children, std::string op)
        : _data(d), _prev(children), _op(op) {}
};
} // namespace forgeml
