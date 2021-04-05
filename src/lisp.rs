use std::{
    collections::HashMap,
    fmt, io,
    ops::{Add, Sub},
    str::FromStr,
    sync::Arc,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nested_add() -> Result<(), LispError> {
        let token_list = tokenize("(+ (+ 1 2) 3)")?;
        let eval = LispExpression::build_from_tokens(&mut token_list.into_iter())?
            .evaluate(&mut LispEnv::default())?;

        println!("{:#?}", &eval);

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

        //println!("{:#?}", &token_list);

        let ast = LispExpression::build_from_tokens(&mut token_list.into_iter())?;

        //println!("{:#?}", &ast);

        let eval = ast.evaluate(&mut LispEnv::default())?;

        println!("{:#?}", &eval);

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
    SymbolNotFound(String),
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
            _ => Err(LispError::RuntimeError(format!(
                "Cannot add {:?} and {:?}",
                self, rhs
            ))),
        }
    }
}

impl<'a> Add for &'a LispLiteral {
    type Output = Result<LispLiteral, LispError>;

    fn add(self, rhs: Self) -> Self::Output {
        match self {
            &LispLiteral::Int(i) => match rhs {
                &LispLiteral::Int(k) => Ok(LispLiteral::Int(i + k)),
                &LispLiteral::Float(k) => Ok(LispLiteral::Float(i as f32 + k)),
                _ => Err(LispError::RuntimeError(format!(
                    "Cannot add {:?} and {:?}: rhs is not Int",
                    self, rhs
                ))),
            },
            &LispLiteral::Float(i) => match rhs {
                &LispLiteral::Float(k) => Ok(LispLiteral::Float(i + k)),
                &LispLiteral::Int(k) => Ok(LispLiteral::Float(i + k as f32)),
                _ => Err(LispError::RuntimeError(format!(
                    "Cannot add {:?} and {:?}: rhs is not Float",
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

pub struct LispEnv {
    procedures: HashMap<
        String,
        Arc<dyn Fn(&mut Self, &mut LispExpression) -> Result<LispExpression, LispError>>,
    >,
    symbols: HashMap<String, LispExpression>,
}

impl Default for LispEnv {
    fn default() -> Self {
        let mut env = Self::new();

        env.add_symbol(
            "t",
            LispExpression::Atom(LispAtom::Literal(LispLiteral::Boolean(true))),
        );
        env.add_symbol(
            "nil",
            LispExpression::Atom(LispAtom::Literal(LispLiteral::Boolean(false))),
        );

        // this is probably horrible but why would anyone ever want to do anything in an interpreted language where you have to check literally everything to add 2 numbers
        env.add_procedure("+", |_env, expr| match expr {
            LispExpression::Atom(LispAtom::Literal(LispLiteral::Float(_))) => Ok(expr.to_owned()),
            LispExpression::Atom(LispAtom::Literal(LispLiteral::Int(_))) => Ok(expr.to_owned()),
            LispExpression::List(list) => {
                if list.is_empty() {
                    Ok(LispExpression::Atom(LispAtom::Literal(LispLiteral::Int(0))))
                } else {
                    Ok(LispExpression::Atom(LispAtom::Literal(
                        list.iter()
                            .map(|expr| match expr {
                                LispExpression::Atom(LispAtom::Literal(lit)) => match lit {
                                    LispLiteral::Float(_) => Ok(lit.to_owned()),
                                    LispLiteral::Int(_) => Ok(lit.to_owned()),
                                    _ => Err(LispError::RuntimeError(
                                        "cant add expressions".to_string(),
                                    )),
                                },
                                _ => {
                                    Err(LispError::RuntimeError("cant add expressions".to_string()))
                                }
                            })
                            .reduce(|a, b| (a? + b?))
                            .unwrap_or(Ok(LispLiteral::Int(0)))?,
                    )))
                }
            }
            _ => Err(LispError::RuntimeError(format!(
                "cannot add expression {:?}",
                expr
            ))),
        });

        env
    }
}

impl LispEnv {
    fn new() -> Self {
        Self {
            procedures: HashMap::new(),
            symbols: HashMap::new(),
        }
    }

    fn add_procedure<F, S>(&mut self, symbol: S, proc: F)
    where
        F: Fn(&mut Self, &mut LispExpression) -> Result<LispExpression, LispError> + 'static,
        S: ToString,
    {
        self.symbols.insert(
            symbol.to_string(),
            LispExpression::Atom(LispAtom::Identifier(symbol.to_string())),
        );

        self.procedures.insert(symbol.to_string(), Arc::new(proc));
    }

    fn add_symbol<S>(&mut self, symbol: S, value: LispExpression)
    where
        S: ToString,
    {
        self.symbols.insert(symbol.to_string(), value);
    }
}

// I think this is equivalent to a Lisp AST node
#[derive(Debug, PartialEq, Clone)]
pub enum LispExpression {
    Atom(LispAtom),
    List(Vec<LispExpression>),
}

impl LispExpression {
    fn new() -> Self {
        Self::List(vec![])
    }

    fn build_from_tokens_internal<I>(token_iter: &mut I) -> Result<Self, LispError>
    where
        I: Iterator<Item = LispToken>,
    {
        let mut node = vec![];

        while let Some(tkn) = token_iter.next() {
            match tkn {
                LispToken::ClosingParens => break,
                LispToken::Atom(atom) => node.push(Self::Atom(atom)),
                LispToken::OpeningParens => {
                    node.push(Self::build_from_tokens_internal(token_iter)?)
                }
            }
        }

        Ok(Self::List(node))
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

    fn evaluate_atom(&self, env: &mut LispEnv) -> Result<Self, LispError> {
        assert!(std::matches!(self, Self::Atom { .. }));

        match self {
            Self::Atom(atom) => match atom {
                LispAtom::Identifier(sym) => Ok(env
                    .symbols
                    .get(sym)
                    .cloned()
                    .ok_or(LispError::SymbolNotFound(sym.clone()))?),
                LispAtom::Literal(lit) => Ok(LispExpression::Atom(LispAtom::Literal(lit.clone()))),
            },
            _ => unreachable!(),
        }
    }

    fn evaluate_list(&self, env: &mut LispEnv) -> Result<Self, LispError> {
        assert!(std::matches!(self, Self::List { .. }));

        match self {
            Self::List(list) => {
                if list.is_empty() {
                    Ok(Self::Atom(LispAtom::Literal(LispLiteral::Boolean(false))))
                } else {
                    if let Self::Atom(LispAtom::Identifier(identifer)) = &list[0] {
                        if identifer == "if" && list.len() == 4 {
                            let test = &list[1];
                            let a = &list[2];
                            let b = &list[3];

                            if test.evaluate(env)?.is_true() {
                                a.evaluate(env)
                            } else {
                                b.evaluate(env)
                            }
                        } else if identifer == "defvar" && list.len() == 3 {
                            let (name, value) = (&list[1], &list[2]);

                            let name = name.evaluate(env)?;
                            let value = value.evaluate(env)?;

                            match name {
                                Self::Atom(LispAtom::Identifier(ref sym_name)) => {
                                    env.add_symbol(sym_name, value);

                                    Ok(name)
                                }
                                _ => Err(LispError::RuntimeError(
                                    "variable name did not evaluate to string identifier"
                                        .to_string(),
                                )),
                            }
                        } else {
                            // this is all the keywords im checking for so far, so this branch is exhausted
                            Err(LispError::Exhausted)
                        }
                    } else {
                        // i dont care about cases where the first expression is not an identifier currently
                        Err(LispError::Exhausted)
                    }
                    .or_else(|_| {
                        let proc = &list[0].evaluate(env)?;

                        let args = list
                            .iter()
                            .skip(1)
                            .map(|exp| exp.evaluate(env))
                            .collect::<Result<Vec<_>, _>>()?;

                        match proc {
                            Self::Atom(LispAtom::Identifier(ref proc_name)) => {
                                match env.procedures.get(proc_name).cloned() {
                                    Some(proc) => proc(env, &mut Self::List(args)),
                                    None => Err(LispError::RuntimeError(
                                        "procedure name did not evaluate to a valid identifier"
                                            .to_string(),
                                    )),
                                }
                            }
                            _ => Err(LispError::RuntimeError(
                                "procedure name did not evaluate to string identifier".to_string(),
                            )),
                        }
                    })
                }
            }
            _ => unreachable!(),
        }
    }

    pub fn evaluate(&self, env: &mut LispEnv) -> Result<Self, LispError> {
        match self {
            Self::Atom(_) => self.evaluate_atom(env),
            Self::List(_) => self.evaluate_list(env),
        }
    }

    pub fn is_nil(&self) -> bool {
        match self {
            Self::List(list) => list.is_empty(),
            Self::Atom(LispAtom::Literal(LispLiteral::Boolean(b))) => b.clone(),
            _ => false,
        }
    }

    pub fn is_true(&self) -> bool {
        !self.is_nil()
    }
}
