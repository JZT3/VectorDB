#pragma once

#include <string>
#include <vector>
#include <Eigen/Dense>

class VectorDatabase; // Forward declaration

class DatabaseSerializer {
public:
    static bool serialize(const VectorDatabase& db, const std::string& filename);
    static bool deserialize(VectorDatabase& db, const std::string& filename);

private:
    static std::string serializeVector(const Eigen::VectorXd& vec);
    static Eigen::VectorXd deserializeVector(const std::string& data);
};