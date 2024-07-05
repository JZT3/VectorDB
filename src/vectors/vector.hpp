#include <Eigen/Dense>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <functional>
#include "LSHIndex.hpp"

class DatabaseSerializer; // Forward declaration

class VectorDatabase {
    public:
        using VectorXd = Eigen::VectorXd;
        
        enum class IndexType { LSH }; // Later to be implemented KD_TREE, VP_TREE, BALL_TREE, HNSW
        enum class DistanceMetric { COSINE }; // Later to be implemented EUCLIDEAN, MANHATTAN, 
        enum class LogLevel { DEBUG, INFO, WARN, ERROR };
        // enum class Mode { SPEED, ACCURACY }; Dynamic modes later to be implemented

        class VectorBuilder 
        {
            public:
                VectorBuilder& setInitialCapacity(size_t capacity) { initialCapacity = capacity; return *this; }
                VectorBuilder& setDimension(size_t dim) { dimension = dim; return *this; }
                VectorBuilder& setDistanceMetric(DistanceMetric metric) { distanceMetric = metric; return *this; }
                VectorBuilder& setStoragePath(const std::string& path) { storagePath = path; return *this; }
                VectorBuilder& setNumberOfHashTables(size_t num) { numberOfHashTables = num; return *this; }
                VectorBuilder& setLSHRadius(double radius) { lshRadius = radius; return *this; }
                //VectorBuilder& setMode(Mode m) { mode = m; return *this; }
                VectorBuilder& setLogLevel(LogLevel level) { logLevel = level; return *this; }
                VectorBuilder& setIndexType(IndexType type) { indexType = type; return *this; }

                VectorDatabase build() const {return VectorDatabase(*this);}

            private:
                friend class VectorDatabase;
                size_t initialCapacity = 1000;
                size_t dimension = 0;
                DistanceMetric distanceMetric = DistanceMetric::COSINE;
                std::string storagePath = "";
                size_t numberOfHashTables = 4;
                double lshRadius = 1.0;
                //Mode mode = Mode::ACCURACY;
                LogLevel logLevel = LogLevel::INFO;
                IndexType indexType = IndexType::LSH;
        };

        VectorDatabase() = delete; // Default Constructor
        VectorDatabase(const VectorDatabase& other) = delete;         // Copy constructor
        VectorDatabase(VectorDatabase&& other) noexcept; // Move constructor
        VectorDatabase& operator=(const VectorDatabase& other) = delete; // Copy assignment operator
        VectorDatabase& operator=(VectorDatabase&& other) noexcept; // Move assignment operator        
        ~VectorDatabase() = default; // Destructor
        

    public: 
        bool addVector(const Eigen::VectorXd& vec); // Add a vector to the database
        VectorXd getVector(size_t index) const; // Retrieve a vector from the database
        bool serialize(const std::string& filename) const; // Serialize the entire database to a file
        bool deserialize(const std::string& filename); // Deserialize from a file into the database
        
    public: //Getters and Setters for serialization
        const std::vector<VectorXd>& getVectors() const { return m_vectors; }
        size_t getNextId() const { return m_nextId; }
        size_t getCapacity() const { return m_capacity; }
        size_t getDimension() const { return m_dimension; }
        DistanceMetric getDistanceMetric() const { return m_distanceMetric; }
        const std::string& getStoragePath() const { return m_storagePath; }
        size_t getNumberOfHashTables() const { return m_numberOfHashTables; }
        double getLshRadius() const { return m_lshRadius; }
        //Mode getMode() const { return m_mode; }
        LogLevel getLogLevel() const { return m_logLevel; }
        IndexType getIndexType() const { return m_indexType; }

        // Add these setter methods for deserialization
        void setNextId(size_t id) { m_nextId = id; }
        void setCapacity(size_t capacity) { m_capacity = capacity; }
        void setDimension(size_t dimension) { m_dimension = dimension; }
        void setDistanceMetric(DistanceMetric metric) { m_distanceMetric = metric; }
        void setStoragePath(const std::string& path) { m_storagePath = path; }
        void setNumberOfHashTables(size_t num) { m_numberOfHashTables = num; }
        void setLshRadius(double radius) { m_lshRadius = radius; }
        //void setMode(Mode mode) { m_mode = mode; }
        void setLogLevel(LogLevel level) { m_logLevel = level; }
        void setIndexType(IndexType type) { m_indexType = type; }

        void clear(); // clear all data

        friend class DatabaseSerializer;


    private:
    // Helper methods for serialization
        std::string serializeVector(const Eigen::VectorXd& vec) const;
        Eigen::VectorXd deserializeVector(const std::string& data) const;
        std::unique_ptr<flann::Index<flann::L2<double>>> m_flannIndex;
        flann::Matrix<double> m_flannDataset;

    private:
    // Private constructor used by Builder
        explicit VectorDatabase(const VectorBuilder& builder);

    // Helper methods for serialization
        std::string serializeVector(const VectorXd& vec) const;
        VectorXd deserializeVector(const std::string& data) const;
        void initializeDistanceFunction();
        void initializeLSHIndex();

    // Member variables
        std::vector<VectorXd> m_vectors;
        std::unordered_map<size_t, size_t> m_idToIndexMap;
        size_t m_nextId;
        size_t m_capacity;
        size_t m_dimension;
        DistanceMetric m_distanceMetric;
        std::string m_storagePath;
        size_t m_numberOfHashTables;
        double m_lshRadius;
        //Mode m_mode;
        LogLevel m_logLevel;
        IndexType m_indexType;
        
        std::function<double(const VectorXd&, const VectorXd&)> m_distanceFunction;

        std::unique_ptr<LSHIndex> m_lshIndex;
        std::shared_ptr<flann::Matrix<double>> m_flannDataset;
};