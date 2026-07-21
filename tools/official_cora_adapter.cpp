#include <CORA/CORA.h>
#include <CORA/CORA_utils.h>
#include <CORA/pyfg_text_parser.h>

#include <Eigen/SVD>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>

namespace {

std::string jsonEscape(const std::string &value) {
  std::string escaped;
  for (const char item : value) {
    switch (item) {
    case '\\':
      escaped += "\\\\";
      break;
    case '"':
      escaped += "\\\"";
      break;
    case '\n':
      escaped += "\\n";
      break;
    case '\r':
      escaped += "\\r";
      break;
    case '\t':
      escaped += "\\t";
      break;
    default:
      escaped += item;
    }
  }
  return escaped;
}

void writeVector(std::ostream &output, const CORA::Matrix &value) {
  output << "[";
  for (Eigen::Index index = 0; index < value.size(); ++index) {
    if (index != 0) {
      output << ",";
    }
    output << value(index);
  }
  output << "]";
}

void writeMatrix(std::ostream &output, const CORA::Matrix &value) {
  output << "[";
  for (Eigen::Index row = 0; row < value.rows(); ++row) {
    if (row != 0) {
      output << ",";
    }
    output << "[";
    for (Eigen::Index column = 0; column < value.cols(); ++column) {
      if (column != 0) {
        output << ",";
      }
      output << value(row, column);
    }
    output << "]";
  }
  output << "]";
}

CORA::Matrix translationFor(const CORA::Symbol &symbol,
                            const CORA::Problem &problem,
                            const CORA::Matrix &solution) {
  return solution.row(problem.getTranslationIdx(symbol));
}

CORA::Matrix rotationFor(const CORA::Symbol &symbol,
                         const CORA::Problem &problem,
                         const CORA::Matrix &solution) {
  const CORA::Index start = problem.getRotationIdx(symbol) * problem.dim();
  return solution
      .block(start, 0, problem.dim(), problem.dim())
      .transpose();
}

int numericalRank(const CORA::Matrix &value) {
  Eigen::JacobiSVD<CORA::Matrix> decomposition(value);
  const auto singular = decomposition.singularValues();
  if (singular.size() == 0 || singular(0) == 0.0) {
    return 0;
  }
  const double tolerance =
      singular(0) * static_cast<double>(std::max(value.rows(), value.cols())) *
      std::numeric_limits<double>::epsilon();
  int rank = 0;
  for (Eigen::Index index = 0; index < singular.size(); ++index) {
    rank += singular(index) > tolerance ? 1 : 0;
  }
  return rank;
}

template <typename SymbolMap>
void writeTranslations(std::ostream &output, const SymbolMap &symbols,
                       const CORA::Problem &problem,
                       const CORA::Matrix &solution) {
  output << "{";
  bool first = true;
  for (const auto &entry : symbols) {
    if (!first) {
      output << ",";
    }
    first = false;
    output << "\"" << jsonEscape(entry.first.string()) << "\":{";
    output << "\"translation\":";
    writeVector(output, translationFor(entry.first, problem, solution));
    output << "}";
  }
  output << "}";
}

void writePoses(std::ostream &output,
                const std::map<CORA::Symbol, int> &symbols,
                const CORA::Problem &problem,
                const CORA::Matrix &solution) {
  output << "{";
  bool first = true;
  for (const auto &entry : symbols) {
    if (!first) {
      output << ",";
    }
    first = false;
    output << "\"" << jsonEscape(entry.first.string()) << "\":{";
    output << "\"translation\":";
    writeVector(output, translationFor(entry.first, problem, solution));
    output << ",\"rotation\":";
    writeMatrix(output, rotationFor(entry.first, problem, solution));
    output << "}";
  }
  output << "}";
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "usage: official_cora_adapter INPUT.pyfg OUTPUT.json SEED"
              << std::endl;
    return 2;
  }
  const std::string input_path = argv[1];
  const std::string output_path = argv[2];
  const unsigned int seed = static_cast<unsigned int>(std::stoul(argv[3]));
  try {
    std::srand(seed);
    CORA::Problem problem = CORA::parsePyfgTextToProblem(input_path);
    problem.updateProblemData();
    const CORA::Matrix initial = problem.getRandomInitialGuess();
    const auto started = std::chrono::steady_clock::now();
    const CORA::CoraResult result = CORA::solveCORA(problem, initial, 10);
    const double solve_time = std::chrono::duration<double>(
                                  std::chrono::steady_clock::now() - started)
                                  .count();
    CORA::Matrix certificate_bootstrap = result.first.x;
    if (problem.getFormulation() == CORA::Formulation::Implicit) {
      certificate_bootstrap =
          problem.getTranslationExplicitSolution(certificate_bootstrap);
    }
    const double certificate_eta =
        std::max(1e-7, std::min(1e-1, result.first.f * 5e-6));
    const CORA::CertResults certificate = problem.certify_solution(
        result.first.x, certificate_eta, 10, certificate_bootstrap);

    std::ofstream output(output_path);
    if (!output) {
      throw std::runtime_error("cannot open result destination");
    }
    output << std::setprecision(17);
    output << "{";
    output << "\"schema_version\":1,";
    output << "\"backend\":\"official_cora\",";
    output << "\"solved\":true,";
    output << "\"certified\":"
           << (certificate.is_certified ? "true" : "false") << ",";
    output << "\"objective\":" << result.first.f << ",";
    output << "\"objective_recomputed\":"
           << problem.evaluateObjective(result.first.x) << ",";
    output << "\"theta\":" << certificate.theta << ",";
    output << "\"certificate_eta\":" << certificate_eta << ",";
    output << "\"certificate_iterations\":" << certificate.num_iters << ",";
    output << "\"dimension\":" << problem.dim() << ",";
    output << "\"relaxation_rank\":" << problem.getRelaxationRank() << ",";
    output << "\"returned_solution_rank\":" << numericalRank(result.first.x)
           << ",";
    output << "\"pose_count\":" << problem.numPoses() << ",";
    output << "\"landmark_count\":" << problem.numLandmarks() << ",";
    output << "\"range_factor_count\":" << problem.numRangeMeasurements()
           << ",";
    output << "\"solve_time_sec\":" << solve_time << ",";
    output << "\"seed\":" << seed << ",";
    output << "\"poses\":";
    writePoses(output, problem.getPoseSymbolMap(), problem, result.first.x);
    output << ",\"landmarks\":";
    writeTranslations(output, problem.getLandmarkSymbolMap(), problem,
                      result.first.x);
    output << "}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "official CORA adapter failed: " << error.what() << std::endl;
    return 1;
  }
}
