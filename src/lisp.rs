use std::{
    collections::HashMap,
    fmt, io,
    ops::{Add, Sub},
    str::FromStr,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ast_builder() -> Result<(), LispError> {
        let token_list = tokenize("(begin (defvar x 10) (+ x 5))")?;

        println!("{:#?}", &token_list);

        let ast = LispExpression::build_from_tokens(&mut token_list.into_iter())?;

        println!("{:#?}", &ast);

        Ok(())
    }

    #[test]
    fn test_tokenizer() -> Result<(), LispError> {
        let token_list = tokenize("(+ 2 2)")?;

        assert_eq!(
            vec![
                LispToken::OpeningParens,
                LispToken::Atom(LispAtom::Identifier("+".to_string())),
                LispToken::Atom(LispAtom::Literal(LispLiteral::Int(2))),
                LispToken::Atom(LispAtom::Literal(LispLiteral::Int(2))),
                LispToken::ClosingParens,
            ],
            token_list
        );

        println!("{:#?}", &token_list);

        let ast = LispExpression::build_from_tokens(&mut token_list.into_iter())?;

        assert_eq!(
            LispExpression {
                atoms: vec![
                    LispAtom::Identifier("+".to_string()),
                    LispAtom::Literal(LispLiteral::Int(2)),
                    LispAtom::Literal(LispLiteral::Int(2)),
                ],
                expressions: vec![],
            },
            ast
        );

        println!("{:#?}", &ast);

        Ok(())
    }
}

#[derive(Debug)]
pub enum LispError {
    Exhausted,
    NotALiteral(String),
    UnexpectedToken(String),
    ExpectedToken(String),
    RuntimeError(String),
    IoError(io::Error),
}

impl From<io::Error> for LispError {
    fn from(error: io::Error) -> Self {
        LispError::IoError(error)
    }
}

impl fmt::Display for LispError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Display not implemented for this Error, use Debug")
    }
}

pub fn tokenize(input: &str) -> Result<Vec<LispToken>, LispError> {
    input
        .replace("(", " ( ")
        .replace(")", " ) ")
        .split_whitespace()
        .map(|s| LispToken::from_str(s).map_err(|_| LispError::UnexpectedToken(s.to_string())))
        .collect()
}

#[derive(Debug, PartialEq, Clone)]
pub enum LispLiteral {
    Int(i64),
    Float(f32),
    String(String),
    Boolean(bool),
}

impl Add for LispLiteral {
    type Output = Result<Self, LispError>;

    fn add(self, rhs: Self) -> Self::Output {
        match self {
            Self::Int(i) => match rhs {
                Self::Int(k) => Ok(Self::Int(i + k)),
                Self::Float(k) => Ok(Self::Float(i as f32 + k)),
                _ => Err(LispError::RuntimeError(format!(
                    "Cannot add {:?} and {:?}: rhs is not Int",
                    self, rhs
                ))),
            },
            Self::Float(i) => match rhs {
                Self::Float(k) => Ok(Self::Float(i + k)),
                Self::Int(k) => Ok(Self::Float(i + k as f32)),
                _ => Err(LispError::RuntimeError(format!(
                    "Cannot add {:?} and {:?}: rhs is not Float",
                    self, rhs
                ))),
            },
            Self::String(ref i) => match rhs {
                Self::String(ref k) => Ok(Self::String(format!("{}{}", i, k))),
                _ => Err(LispError::RuntimeError(format!(
                    "Cannot add {:?} and {:?}: rhs is not String",
                    self, rhs
                ))),
            },
            _ => Err(LispError::RuntimeError(format!(
                "Cannot add {:?} and {:?}",
                self, rhs
            ))),
        }
    }
}

impl Sub for LispLiteral {
    type Output = Result<Self, LispError>;

    fn sub(self, rhs: Self) -> Self::Output {
        match self {
            Self::Int(i) => match rhs {
                Self::Int(k) => Ok(Self::Int(i - k)),
                Self::Float(k) => Ok(Self::Float(i as f32 - k)),
                _ => Err(LispError::RuntimeError(format!(
                    "Cannot subtract {:?} and {:?}: rhs is not Int",
                    self, rhs
                ))),
            },
            Self::Float(i) => match rhs {
                Self::Float(k) => Ok(Self::Float(i - k)),
                Self::Int(k) => Ok(Self::Float(i - k as f32)),
                _ => Err(LispError::RuntimeError(format!(
                    "Cannot subtract{:?} and {:?}: rhs is not Float",
                    self, rhs
                ))),
            },
            _ => Err(LispError::RuntimeError(format!(
                "Cannot subtract {:?} and {:?}",
                self, rhs
            ))),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum LispAtom {
    Identifier(String),
    Literal(LispLiteral),
}

#[derive(Debug, PartialEq, Clone)]
pub enum LispToken {
    OpeningParens,
    Atom(LispAtom),
    ClosingParens,
}

impl FromStr for LispToken {
    type Err = LispError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "(" => Ok(Self::OpeningParens),
            ")" => Ok(Self::ClosingParens),
            s => Ok(Self::Atom(LispAtom::from_str(s)?)),
        }
    }
}

impl FromStr for LispAtom {
    type Err = LispError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match LispLiteral::from_str(s) {
            Ok(lit) => Ok(Self::Literal(lit)),
            Err(LispError::NotALiteral(s)) => Ok(Self::Identifier(s)),
            Err(err) => Err(err),
        }
    }
}

impl FromStr for LispLiteral {
    type Err = LispError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.parse::<i64>()
            .ok()
            .and_then(|i| Some(Self::Int(i)))
            .or_else(|| s.parse::<f32>().ok().and_then(|f| Some(Self::Float(f))))
            .or_else(|| {
                if s == "t" {
                    Some(Self::Boolean(true))
                } else if s == "nil" {
                    Some(Self::Boolean(false))
                } else {
                    None
                }
            })
            .or_else(|| {
                if s.starts_with('"') && s.ends_with('"') {
                    Some(Self::String(s.trim_matches('"').to_string()))
                } else {
                    None
                }
            })
            .ok_or(LispError::NotALiteral(s.to_string()))
    }
}

pub struct LispEnv<I>
where
    I: Iterator<Item = LispAtom>,
{
    procedures: HashMap<String, Box<dyn Fn(&mut Self, &mut I) -> Result<LispAtom, LispError>>>,
    symbols: HashMap<String, LispLiteral>,
}

impl<I> Default for LispEnv<I>
where
    I: Iterator<Item = LispAtom>,
{
    fn default() -> Self {
        let mut env = Self::new();

        env.add_symbol("t", LispLiteral::Boolean(true));
        env.add_symbol("nil", LispLiteral::Boolean(false));

        env.add_procedure("+", |_env, list| {
            Ok(LispAtom::Literal(
                list.map(|atom| match atom {
                    LispAtom::Literal(lit) => Ok(lit),
                    _ => Err(LispError::RuntimeError(
                        "cannot add non literals".to_string(),
                    )),
                })
                .reduce(|a, b| a? + b?)
                .unwrap_or(Ok(LispLiteral::Int(0)))?,
            ))
        });

        env
    }
}

impl<I> LispEnv<I>
where
    I: Iterator<Item = LispAtom>,
{
    fn new() -> Self {
        Self {
            procedures: HashMap::new(),
            symbols: HashMap::new(),
        }
    }

    fn add_procedure<F, S>(&mut self, symbol: S, proc: F)
    where
        F: Fn(&mut Self, &mut I) -> Result<LispAtom, LispError> + 'static,
        S: ToString,
    {
        self.procedures.insert(symbol.to_string(), Box::new(proc));
    }

    fn add_symbol<S>(&mut self, symbol: S, value: LispLiteral)
    where
        S: ToString,
    {
        self.symbols.insert(symbol.to_string(), value);
    }
}

// I think this is equivalent to a Lisp AST node
#[derive(Debug, PartialEq, Clone)]
pub struct LispExpression {
    atoms: Vec<LispAtom>,
    expressions: Vec<LispExpression>,
}

impl LispExpression {
    fn new() -> Self {
        Self {
            atoms: vec![],
            expressions: vec![],
        }
    }

    fn build_from_tokens_internal<I>(token_iter: &mut I) -> Result<Self, LispError>
    where
        I: Iterator<Item = LispToken>,
    {
        let mut ast = Self::new();

        while let Some(tkn) = token_iter.next() {
            match tkn {
                LispToken::ClosingParens => break,
                LispToken::Atom(atom) => ast.atoms.push(atom),
                LispToken::OpeningParens => ast
                    .expressions
                    .push(Self::build_from_tokens_internal(token_iter)?),
            }
        }

        Ok(ast)
    }

    pub fn build_from_tokens<I>(token_iter: &mut I) -> Result<Self, LispError>
    where
        I: Iterator<Item = LispToken>,
    {
        match token_iter.next() {
            Some(LispToken::OpeningParens) => Self::build_from_tokens_internal(token_iter),
            _ => Err(LispError::ExpectedToken("(".to_string())),
        }
    }

    pub fn is_nil(&self) -> bool {
        self.atoms.is_empty() && self.expressions.is_empty()
    }

    pub fn is_true(&self) -> bool {
        !self.is_nil()
    }
}
