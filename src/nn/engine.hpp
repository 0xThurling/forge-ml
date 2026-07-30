#include <functional>
#include <memory>
#include <string>
#include <vector>

struct Node {
  double data = 0.0;
  double grad = 0.0;
  std::string op;
  std::vector<std::shared_ptr<Node>> prev;
  std::function<void()> backward = [] {};
};

namespace forgeml {
// Stores scalar values and it's gradient
class Value {
  std::shared_ptr<Node> n_;

public:
  Value() : n_(std::make_shared<Node>()) {}
};
} // namespace forgeml
