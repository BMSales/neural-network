{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  name = "dev-shell";
  buildInputs = with pkgs.python313Packages; [
    numpy
    matplotlib
  ];
}
