// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "InfengMetal",
    platforms: [.macOS(.v26)],
    products: [.library(name: "InfengMetal", type: .dynamic, targets: ["InfengMetal"])],
    targets: [.target(name: "InfengMetal", linkerSettings: [.linkedFramework("Metal")])]
)
